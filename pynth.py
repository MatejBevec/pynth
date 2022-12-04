import os
import sys
import math
import numbers

import numpy as np
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt
from graphviz import Digraph

SR = 48000 # TEMP 
CHUNK = 50000 # TIME CHUNK LEN IN SAMPLES
MAXDEPTH = 300
time = 0 # CURRENT CHUNK ON THE CLOCK


class Module():

    def __init__(self):
        self.enabled = True
        self._ins = None
        self._out = None

    def __repr__(self):
        pass

    def _next(self):
        # advance chunk

        # OPTION 1:
        # if all outs have called, advance
        # problem: some outs can be dead ends, not general

        # OPTION 2: 
        # a global (or player node's) clock advances all nodes
        # problem: global seems wrong somehow but is it? -> a real analog synth, runs on a "global clock", no?
            # SUB A: **   time is simply a global var that _chunk uses, no explicit advancing
            #   problem: some modules need to access history longer than a chunk -> specify length in _chunk?
            #   or: also pass time at _chunk
            # SUB B: *   all nodes are stored, and _next is called on all by the clock
            #   problem: those modules have to handle storing of enough chunks
            # SUB C:    _next calls are backpropagated from player, problem: dead end nodes are not updated

        
        pass

    def __add__(self, other):
        if not isinstance(other, Module):
            other = Wave(other)
        return Add(self, other)

    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if not isinstance(other, Module):
            other = Wave(other)
        return Mul(self, other)

    def __rmul__(self, other):
        return self * other

    def play(self, dur, live=False):
        if not live:
            data = self.eval(dur)
            sd.play(data, SR)
            sd.wait()
            return
        for ch, smp in enumerate(range(0, dur, CHUNK)):
            sd.play(self._chunk(ch, 1, 0), SR)
            sd.wait()

    def eval(self, dur):
        data = []
        for ch, smp in enumerate(range(0, dur, CHUNK)):
            data.append(self._chunk(ch, 1, 0))
        return np.concatenate(data)[:dur]

    def _compute():
        pass

    def _chunk(self, t, k=1, d=0):
        if d > MAXDEPTH:
            return np.zeros(k*CHUNK)
        if not hasattr(self, "_mem"):
            self._mem = {}
        if (t, k) in self._mem:
            return self._mem[(t, k)]
        chunk = self._compute(t, k, d)
        #print(type(chunk))
        assert len(chunk) == k*CHUNK
        self._mem[(t, k)] = chunk
        return chunk


class Wave(Module):

    def __init__(self, data):
        super().__init__()
        if isinstance(data, list):
            data = np.array(data)
        self.data = data
        self.chunk = 0 # probably not the best way
        self.start = 0

    def _compute(self, t, k=1, d=0):
        reqst, end = (t+1-k)*CHUNK, (t+1)*CHUNK
        st, neg = max(reqst, 0), min(reqst, 0)
        chunk = None
        if isinstance(self.data, numbers.Number):
            chunk = np.full((end - st,), self.data)
        if isinstance(self.data, np.ndarray):
            l = len(self.data)
            chunk = self.data[min(st, l) : min(end, l)]
            chunk = np.pad(chunk, (0, end-st - len(chunk)))
        if callable(self.data):
            chunk = self.data(np.linspace(st/SR, end/SR, end-st))
        #print(f"{d} Wave ({id(self)}): ", np.pad(chunk, (-neg, 0)).shape)
        chunk = np.pad(chunk, (-neg, 0))
        assert len(chunk) == k*CHUNK
        return chunk


class Sin(Wave):

    def __init__(self, freq, amp=1.0, phase=0.0):
        super().__init__(None)
        self.freq = freq
        self.amp = amp
        self.phase = phase
        self.data = lambda x: amp * np.sin(x*2*math.pi*self.freq + self.phase)

        
    
class Add(Module):

    def __init__(self, ina=None, inb=None):
        super().__init__()
        self.ina = ina
        self.inb = inb
        self._ins = set([ina, inb])
        if ina: ina._out = self
        if inb: inb._out = self

    def _compute(self, t, k=1, d=0):
        a = self.ina._chunk(t, k, d+1)
        b = self.inb._chunk(t, k, d+1)
        #print(f"{d} Add ({id(self)}): ", (a+b).shape)
        chunk = a + b; assert len(chunk) == k*CHUNK
        return chunk


class Mul(Module):

    def __init__(self, ina=None, inb=None):
        super().__init__()
        self.ina = ina
        self.inb = inb
        self._ins = set([ina, inb]) # TODO: this doesn't get called when manually adding ins
        if ina: ina._out = self
        if inb: inb._out = self
    
    def _compute(self, t, k=1, d=0):
        a = self.ina._chunk(t, k, d+1)
        b = self.inb._chunk(t, k, d+1)
        chunk = a * b; assert len(chunk) == k*CHUNK
        return a * b


class Lowpass(Module):
    
    # TODO

    def __init__(self, ina=None, cutoff=500, spread=200):
        self.ina = ina
        self._ins = set([ina])
        if ina: ina._out = self
        
        self.window = max(20000, CHUNK)
        self.cutoff = cutoff
        ct_idx = int(cutoff * self.window / SR)
        self.filt = np.ones(self.window)
        self.filt[ct_idx:ct_idx+spread] = np.linspace(1, 0, spread)
        self.filt[ct_idx+spread:] = 0

    def _func(self, x):
        # TODO: incompatible with small chunks, fix it
        freqs = np.fft.fft(x)
        freqs = self.filt * freqs
        return np.real(np.fft.ifft(freqs))

    def _compute(self, t, k=1, d=0):
        self.window = k*CHUNK # TEMP
        reqk =  max(int(self.window/CHUNK) + 1, k)
        x = self.ina._chunk(t, reqk, d+1)
        chunk = self._func(x[-self.window:])
        #print(f"{d} Lowpass ({id(self)}): ", chunk.shape, k*CHUNK)
        chunk = chunk[-k*CHUNK:]
        assert len(chunk) == k*CHUNK 
        return chunk


class Delay(Module):

    def __init__(self, ina=None, delay=0.01):
        self.ina = ina
        self._ins = set([ina])
        if ina: ina._out = self
        self.delay = delay
        self.dsmp = int(delay * SR)
        self.dch = int(self.dsmp / CHUNK) + 1
        #self.prevch = np.zeros(CHUNK) # TODO: sloppy, put this logic in super class
    
    def _compute(self, t, k=1, d=0):
        chunk = np.zeros(k*CHUNK)
        hist = self.ina._chunk(t, k + self.dch, d+1)
        chunk = hist[-k*CHUNK -self.dsmp : -self.dsmp]
        assert len(chunk) == k*CHUNK 
        return chunk

    


    


def showsound(module, x1=0, x2=30000, sec=False):
    if sec:
        x1, x2 = int(x1*SR), int(x2*SR)
    t = int(x2/CHUNK) + 1
    k = int((x2-x1)/CHUNK) + 3
    t1, t2 = (t+1-k)*CHUNK, (t+1)*CHUNK
    ch = module._chunk(t, k, 0)
    x = np.linspace(t1, t2, (t2-t1))
    if sec:
        x /= SR
        x1, x2 = x1/SR, x2/SR
    plt.plot(x, ch)
    plt.xlim((x1, x2))
    plt.show()

def showsound2(module, x1=0, x2=30000, sec=False):
    if sec:
        x1, x2 = int(x1*SR), int(x2*SR)
    data = module.eval(x2)
    x = np.linspace(0, x2, x2)
    if sec:
        x /= SR
        x1, x2 = x1/SR, x2/SR
    plt.plot(x, data)
    plt.xlim((x1, x2))
    plt.show()




def trace(root):
    root._isroot = True
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            if v._ins:
                for child in v._ins:
                    edges.add((child, v))
                    build(child)
    build(root)
    return nodes, edges

def drawgraph(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        # "{ data %.4f | grad %.4f }" % (n.data, n.grad)
        isroot = hasattr(n, "_isroot") and n._isroot
        color = 'black'
        if isinstance(n, Wave): color = 'blue'
        if isroot: color = 'red'
        label = n.__class__.__name__
        if isinstance(n, Add): label = '+'
        if isinstance(n, Mul): label = "*"
        dot.node(name=str(id(n)), label=label, shape='record', color=color)
        # if n._op:
        #     dot.node(name=str(id(n)) + n._op, label=n._op)
        #     dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2))) # + n2._op
    
    return dot



if __name__ == "__main__":

    #wav = Wave(np.linspace(0, 5, 50))

    # saw = Sin(100)
    # for i in range(5):
    #     saw += Sin(i*100)

    # mod  = Sin(1)
    # out = mod * saw
    # out = Lowpass(out)

    # square = Sin(30)
    # for i in range(1, 11, 2):
    #     square += Sin(i*40)

    # sqmod = Sin(2)
    # for i in range(1, 11, 2):
    #     sqmod += Sin(i*0.5)
    
    # out += square * sqmod

    #noise = Wave(np.random.rand(4000)-0.5)
    # lowpass = Lowpass(None, cutoff=500, spread=300)

    # add = Sin(300)
    # add += Lowpass(Delay(add, delay=0.001))
    # add += Lowpass(Delay(add, delay=0.002))
    # add += Lowpass(Delay(add, delay=0.003))
    # add += Lowpass(Delay(add, delay=0.004))

    # out = add


    #out = noise +  Sin(300) * Sin(2)

    # Quazi carplus
    # noise = Wave(np.random.rand(4000)-0.5)
    # out = noise
    # for i in range(1, 30):
    #     val = np.array([((10-i)/10)])[0]
    #     out +=  val * Delay(out, delay=0.01*i)


    # AMP MODULATION

    out = Sin(50) * Sin(1)

    g = drawgraph(out); g.render("gout", view=True)
    showsound2(out, x2=24000)
    out.play(50000)


    # SAW WAVE FROM SINS
    
    out = Sin(200)
    for i in range(2, 10, 2):
        out += Sin(200 * i)

    g = drawgraph(out); g.render("gout", view=True)
    showsound2(out, x2=1000)
    out.play(50000)


    # KARPLUS STRONG (without lowpass)

    noise = Wave(np.random.rand(4000)-0.5)
    delay = Delay(None, delay=0.005)
    add = noise + delay
    delay.ina = 0.95 * add
    delay._ins = set([delay.ina])
    out = add
    
    g = drawgraph(out); g.render("gout", view=True)
    showsound2(out, x2=50000)
    out.play(50000)

    



# Something like this (modules like jsyn but cleaner):
    # sin = pynth.sin(400, 1)
    # saw = pynth.saw(200, 0.5)
    # shaped = pynth.waveshape(sin + saw, shape)
    # out = pynth.eq(shaped, bands=3, values=[0.9, 0.9, 1.1])
    # out.play()
# Where waves can either have a start and end time or not

# Or like autograd?:
    # def tr():
        # sin = pynth.sin(400, 1)
        # saw = pynth.saw(200, 0.5)
        # shaped = pynth.waveshape(sin + saw, shape)
        # out = pynth.eq(shaped, bands=3, values=[0.9, 0.9, 1.1])
    # tr().play()


