import os
import sys
import math
import numbers
import inspect
import time

import numpy as np
import scipy as sp 
import librosa
import sounddevice as sd
import matplotlib
#matplotlib.use("webagg")
import matplotlib.pyplot as plt
from graphviz import Digraph

SR = 48000 # TEMP 
CHUNK = 1000 # TIME CHUNK LEN IN SAMPLES
MAXDEPTH = 30
gtime = 0 # CURRENT CHUNK ON THE CLOCK - TEMP
globnodes = []


# every module has _ins and _outdata 
# when a module is called, it takes _ins[i]._outdata and computes its own _outdata

# module types are:
# generator ->
# filter -> takes _ins, produces _outdata, implements _filter(x, y, zi)
#       * linear -> needs to only init _a and _b or _impres, idk if as attributes or function
#       * frequency -> frequency domain filter, probably implements _freqfilt(X) where X is the spectrum with standardized axes



# this is done from waves to root (output) after a topo sort of nodes
# because there can be cycles, topo sort will fail
#   -> try computing nodes anyways and if any dont have ready inputs, just assume np.zeros() ?
# the clock is local to root? -> and passed around in _proc


# DECISIONS:
# prioritize DRY/abstraction or atomic modules?
# should arithmetic/logic operations accept 2 inputs or any number of
# -> or have "composite" modules

# THE BIG ISSUE:
# use small chunks (smaller than min delay), make loops easy, sacrifice speed
# or support larger chunks, make loops resolutions really complex, make use of numpy speedups
# VERDICT: FIRST OPTION FOR NOW


# TODO: TRIGGERED WAVES like an Envelope should be generalized somehow
    # TODO: any submodule should only implement the basic "response", time shifting, memory across chunks
    # etc. should be generalized

# TODO: MODULATING PARAMETERS
    # TODO: either modules like Delay, Oscillators etc. take a "control" input
    # TODO: or there is a wrapped Modulate, that can change any parameters of the wrapped module

# TODO: INPUTS - how should they be passed in and modified later
    # TODO: modify via controlled method, so everything is recomputed properly
    # TODO: Compose - control input setting - the composed module needs to be updated on input changes

# TODO: consider using factory methods like sin(), then saw() can return a special Triangle

# TODO: auto normalize signals

# TODO: if we assume tiny chunks, some modules can probably be simplified


def toposort(out):
    topo = []
    visited = set()
    def _build(v):
        if v not in visited:
                visited.add(v)
                for child in v.ins.values():
                    _build(child)
                topo.append(v)
    _build(out)
    return topo


class Module():

    def __init__(self):
        # self.ins = {}
        # undefined inputs are always treated as zero signals
        if not hasattr(self, "ins"): self.ins = {}
        self.indata = {inn: np.zeros(CHUNK) for inn in self.ins}
        self.outdata = np.zeros(CHUNK)
        self._mem = {}
        self._ready = False

        self.i = len(globnodes)
        globnodes.append(self.i)

    def __repr__(self):
        return f"{self.i}: {self.__class__.__name__}"

    def play(self, dur, live=False):
        if not live:
            data = self.eval(dur)
            sd.play(data, SR)
            sd.wait()
            return
        
        global gtime; gtime = 0
        def _callback(outdata, frames, t, status):
            global gtime
            outdata[:] = self.proc(gtime)
            gtime += CHUNK
        with sd.OutputStream(samplerate=SR, blocksize=CHUNK, callback=_callback):
            sd.sleep(int(dur/SR * 1000))

    def eval(self, dur):
        """Process [0, dur/CHUNK] chunks for all modules and concat root output."""

        data = []
        for t, smp in enumerate(range(0, dur, CHUNK)):
            self.proc(t)
            data.append(self.outdata)
        return np.concatenate(data)[:dur]

    def _compute(self, t):
        """Compute the next chunk.
        
        Args:
            t: time (samples) at the beginning of this chunk
            d: depth, i.e. how many times this has been called at this t, to stop infinite loops
        """
        pass

    def _filter(self, x, y, zi):
        """Compute filtered values for this chunk.
        
        Args:
            x: unfiltered values for interval [t, t+CHUNK]
            y: filtered values for interval [t, t+CHUNK]
            zi: filter memory from previous chunks
        """
        pass

    def proc(self, t):
        """Process one chunk for all upstream modules."""
        topo = toposort(self)

        #print(topo)

        # Lets try the acyclical case first
        for node in topo:
            node._proc(t, 0)

        # for i in range(50):
        #     for node in topo:
        #         node._proc(t, 0)
            


    def _proc(self, t, d=0):
        """Process one chunk for this module."""
        #print(f"_proc on {self}")

        for inn in self.ins:
            if self.ins[inn] is not None:
                self.indata[inn] = self.ins[inn].outdata
        if d > MAXDEPTH:
            return

        chunk = self._compute(t, d)
        self.outdata = chunk  

    # OPERATIONS

    def __add__(self, other):
        if not isinstance(other, Module):
            other = Wave(other)
        return Add(self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if not isinstance(other, Module):
            other = Wave(other)
        return Sub(self, other)
    
    def __rsub__(self, other):
        return self - other
    
    def __mul__(self, other):
        if not isinstance(other, Module):
            other = Wave(other)
        return Mul(self, other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if not isinstance(other, Module):
            other = Wave(other)
        return Div(self, other)

    def __rtruediv__(self, other):
        if not isinstance(other, Module):
            other = Wave(other)
        return other / self

    def __pow__(self, other):
        if not isinstance(other, Module):
            other = Wave(other)
        return Pow(self, other)

    def __rpow__(self, other):
        if not isinstance(other, Module):
            other = Wave(other)
        return other ** self

    def __gt__(self, other):
        if not isinstance(other, Module):
            other = Wave(other)
        return Compare(self, other)

    def __lt__(self, other):
        if not isinstance(other, Module):
            other = Wave(other)
        return other > self

    def __rshift__(self, other):
        assert isinstance(other, numbers.Number)
        return Delay(self, delay=other)



class Wave(Module):
    """A signal generator module"""

    def __init__(self, data, pad=0):
        if isinstance(data, list):
            data = np.array(data)
        if data is not None: self.data = data
        self._pad = pad
        super().__init__()

    def _compute(self, t, d=0):
        reqst, end = (t)*CHUNK, (t+1)*CHUNK
        st, neg = max(reqst, 0), min(reqst, 0)
        chunk = None
        if isinstance(self.data, numbers.Number):
            chunk = np.full((end - st,), self.data)
        if isinstance(self.data, np.ndarray):
            l = len(self.data)
            chunk = self.data[min(st, l) : min(end, l)]
            chunk = np.pad(chunk, (0, end-st - len(chunk)), constant_values=self._pad)
        if callable(self.data):
            chunk = self.data(np.linspace(st/SR, end/SR, end-st))
        chunk = np.pad(chunk, (-neg, 0))
        assert len(chunk) == CHUNK
        return chunk


class Ramp(Wave):
    """Gradual interpolation between two signal values"""

    def __init__(self, dur, range=[0,1], type="linear"):
        # TODO: other types
        self.dur = dur
        self.range = range
        self.type = type
        st, end = int(dur[0]*SR), int(dur[1]*SR)
        ramp = np.linspace(range[0], range[1], end-st)
        data = np.zeros(end)
        data[st:end] = ramp
        super().__init__(data, pad=data[-1])

class Pulse(Wave):
    """Single pulse of given duration"""

    def __init__(self, dur):
        self.dur = dur
        st, end = int(dur[0]*SR), int(dur[1]*SR)
        data = np.zeros(end)
        data[st:end] = 1
        super().__init__(data, pad=0)

class Pulses(Wave):
    """Continuous pulse train"""

    def __init__(self, w=1/SR, T=0.3):
        self.w = w
        self.T = T
        super().__init__(None)

    def data(self, t):
        data = np.zeros(len(t))
        ts = (t*SR).astype(int)
        data[ts % (self.T*SR) < self.w*SR] = 1
        return data


def get_crossings(a):
    mask = (a > 0.5)
    shifted = np.ones(len(mask))
    shifted[1:] = mask[:-1]
    imp = np.logical_and(mask, np.logical_not(shifted))
    loc = np.nonzero(imp)[0]
    return loc


class Envelope(Wave):

    # TODO: ADSR not just AD

    def __init__(self, ina, durs=(4000, 2000, 0, 10000), suslvl=0.7, thr=0.5):
        self.ins = {"a": ina}
        self.durs = durs
        self.suslvl = suslvl
        self.thr = thr
        att, dec, sus, rel, = durs
        # sus is the sustain duration if key is released immediately

        attack = np.linspace(0, 1, att)
        decay = np.linspace(1, 0, dec)**2 * (1-suslvl) + suslvl
        #sustain = np.ones(20000) * suslvl
        release = np.linspace(1, 0, rel)**2 * suslvl

        self.adenv = np.concatenate([np.linspace(0, 1, att), np.linspace(1, 0, rel)**2])

        self._state = 0 # 0-idle, 1-pressed, 2-released
        self._t = 0 # time since state change

        self.history = np.zeros(CHUNK)

        super().__init__(None)

    def data(self, t):
        # BODGE
        # BUG: what if the press is aligned with segment start
        a = self.indata["a"]
        out = np.zeros(len(t))
        out[:len(self.history)] = self.history[:len(out)]
        self.history = self.history[len(out):]
        loc = get_crossings(a)
        for i in loc:
            inlen = min(len(self.adenv), len(out)-i)
            out[i: i+inlen] = self.adenv[:inlen]
            self.history = self.adenv[inlen:]

        return out


class Sequencer(Wave):
    # TODO: how to handle possible repeat calls

    def __init__(self, ina, sequence):
        self.ins = {"a": ina}
        self.sequence = sequence
        self.state = 0
        super().__init__(None)

    def data(self, t):
        a = self.indata["a"]
        out = np.zeros(len(t))
        loc = get_crossings(a)
        print(loc)
        prev = 0
        for i in loc:
            print(prev, i)
            out[prev:i] = float(self.sequence[self.state])
            out[i-1] = 0.0
            self.state = (self.state + 1) % len(self.sequence)
            prev = i
        out[prev:] = float(self.sequence[self.state])
        return out


# BASIC OSCILLATORS

# TODO: consider removing amp and phase, it can be done with Mul and Delay

class Sin(Wave):

    def __init__(self, freq, amp=1.0, phase=0.0):
        self.freq = freq
        self.amp = amp
        self.phase = phase
        data = lambda t: self.amp * np.sin(t*2*math.pi*self.freq + self.phase)
        super().__init__(data)

class Triangle(Wave):

    def __init__(self, freq, amp=1.0, ratio=0.5):
        super().__init__(None)
        self.freq = freq
        self.amp = amp
        self.ratio = ratio
        p = 1/self.freq
        rtime = p*ratio
        self.data = lambda t: (t%p <= rtime) * ((t%p)  / rtime - 1) -  (t%p > rtime ) * (((t%p) - rtime) / (p - rtime))

class Saw(Wave):

    def __init__(self, freq, amp=1.0):
        super().__init__(None)
        self.freq = freq
        self.amp = amp
        self.data = lambda t: self.amp * 2 * (t*self.freq - np.floor(0.5 + t*self.freq))

class Square(Wave):

    def __init__(self, freq, amp=1.0):
        super().__init__(None)
        self.freq = freq
        self.amp = amp
        self.data = lambda t: self.amp * np.sign(np.sin(t*2*math.pi*self.freq))


# OPERATIONS

class TwoOp(Module):
    """An arithmetic or logic operation over two inputs."""

    def __init__(self, ina=None, inb=None):
        self.ins = {"a": ina, "b": inb}
        super().__init__()
    
class Add(TwoOp):

    def _compute(self, t, d=0):
        return self.indata["a"] + self.indata["b"]

class Sub(TwoOp):

    def _compute(self, t, d=0):
        return self.indata["a"] - self.indata["b"]

class Mul(TwoOp):

    def _compute(self, t, d=0):
        return self.indata["a"] * self.indata["b"]

class Div(TwoOp):

    def _compute(self, t, d=0):
        return self.indata["a"] / self.indata["b"]

class Pow(TwoOp):

    def _compute(self, t, d=0):
        #print(type(self.ins["a"]), type(self.ins["b"]))
        return self.indata["a"] ** self.indata["b"]

class Compare(TwoOp):

    def _compute(self, t, d=0):
        return (self.indata["a"] > self.indata["b"]).astype(float)



class Delay(Module):

    def __init__(self, ina=None, delay=0.005):
       
        self.ins = {"a": ina}
        self.delay = delay
        dsamp = int(delay * SR)
        self.b = np.zeros(dsamp + 1); self.b[dsamp] = 1
        self.a = [1]
        self.zi = np.zeros(dsamp)
        super().__init__()
    
    def _compute(self, t, d=0):
        x = self.indata["a"]
        #print(self.b.shape, len(self.a), x.shape, self.zi.shape)
        y, zf = sp.signal.lfilter(self.b, self.a, x, zi=self.zi)
        self.zi = zf
        return y

class Lowpass(Module):
    # TEMP: moving average lowpass

    def __init__(self, ina=None, incontrol=None, cutoff=500):
        self.ins = {"a": ina, "control": incontrol}
        self.cutoff = cutoff
        M = int(SR/cutoff)
        self.b = np.ones(M)/M
        self.a = [1]
        self.zi = np.zeros(M-1)
        super().__init__()

    def setparam(self):
        control = self.indata["control"]
        maxcut = 5000
        mincut = 500
        cutoff = control[0]*maxcut + (1-control[0])*mincut
        print(control[2])
        M = int(SR/cutoff)
        b = np.ones(M)/M
        zi = self.zi[-(M-1):]
        topad = max(M-1 - len(zi), 0)
        zi = np.pad(zi, (topad, 0))
        self.zi = zi
        self.b = b

    def _compute(self, t, d=0):
        if self.ins["control"] is not None:
            self.setparam()
        x = self.indata["a"]
        y, zf = sp.signal.lfilter(self.b, self.a, x, zi=self.zi)
        self.zi = zf
        return y





# COMPOSED MODULES

# TODO: cleaner

# TODO: compose() should accept parameters too

def compose(factory, label=None):
    arginfo = inspect.getfullargspec(factory)[0]
    def constructor(*args):
        assert len(args) == len(arginfo)
        out = Compose()
        out.ins = {argn: arg for argn, arg in zip(arginfo, args)}
        out.module = factory(*args)
        out.__init__()
        if label is not None: out.label = label
        return out
    return constructor


class Compose(Module):

    def __init__(self):
        # set self.ins and self.module
        super().__init__()
        pass

    def _compute(self, t, d=0):
        return self.module._compute(t, d)

    def _proc(self, t, d=0):
        topo = toposort(self.module)
        for node in topo:
            node._proc(t, 0)
        self.outdata = self.module.outdata


Crossfade = compose( lambda a, b, control:  a*control + b*(1-control) , "Crossfade")

Latch = compose( lambda a, control: a * (control > 0) , "Latch")

Crosstrigger = compose( lambda a:  (a > 0) * (((a > 0) >> 1/SR) < 0.5) , "Crosstrigger")



# class Crossfader(Compose):

#     def __init__(self, ina, inb, incontrol):
#         self.ins = {"a": ina, "b": inb, "control": incontrol}
#         self.module = ina*incontrol + inb*(1-incontrol)



    
def showsound(module, x1=0, x2=30000, sec=False):
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
            for child in v.ins.values():
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

# TODO: cleaner

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
        color = 'black'; fontcolor = 'black'
        if isinstance(n, Wave): color = 'blue'
        if isroot: color = 'red'
        label = n.__class__.__name__

        # TODO: put this logic in the classes

        if isinstance(n, Add): label = '+'
        if isinstance(n, Mul): label = "*"
        if isinstance(n, Wave) and isinstance(n.data, numbers.Number):
            label = str(round(n.data, 3))
            color = 'gainsboro'
            fontcolor = 'gainsboro'
        if hasattr(n, "label"): label = n.label
        dot.node(name=str(id(n)), label=label, shape='record', color=color, fontcolor=fontcolor)
        # if n._op:
        #     dot.node(name=str(id(n)) + n._op, label=n._op)
        #     dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        color = 'black'
        if isinstance(n1, Wave) and isinstance(n1.data, numbers.Number) or \
            isinstance(n2, Wave) and isinstance(n2.data, numbers.Number):
            color = 'gainsboro'
        dot.edge(str(id(n1)), str(id(n2)), color=color) # + n2._op
    
    return dot



if __name__ == "__main__":

    # MUSIC?

    # gate = Wave(np.linspace(-1, 1, 50000))

    # impulses = Pulses(1/SR, 0.5)

    # ramp = Ramp((0, 16)) ** 2
    # hum = Sin(100) * ramp * 0.3

    # env = Envelope(impulses >> 0.01, durs=(3000, 0, 0, 12000))
    # voice = Square(50) + Saw(50)

    # out = env * voice +hum

    # mask = Pulse((8, 24))
    # voice2 = Sin(1000) * Sin(4) >> 0.01
    
    # out += mask * voice2

    # ramp2 = Ramp((8, 16)) ** 2
    # out += Sin(30) * ramp2

    # mask2 = Pulse((24, 40))
    # env2 = Envelope(impulses >> 0.011, durs=(1000, 0, 0, 6000))
    # newkick = Sin(50)*2 * env2 * mask2
    # out = 0.7*out + newkick + Sin(4000)*0.2 * mask2


    # FILTER ENVELOPE

    # impulses = Pulses(1/SR, 1)
    # env = Envelope(impulses, durs=(3000, 0, 0, 12000))
    # voice = Saw(200)
    # lowpass = Lowpass(voice, env)

    # print("hello")
    # ramp = Wave(np.linspace(0, 1, 50000))
    # print("world")
    # out = Lowpass(Saw(200), ramp)
    # print("fasfdas")


    # SEQUENCER

    # clock = Square(4)
    # seq = Sequencer(clock, [1, 0, 1, 0, 1, 1, 1, 0]) >> 0.01
    # env = Envelope(seq, durs=(1000, 0, 0, 6000))

    # out = env * Saw(200) * Sin(50)
    # out = Highpass(out)

    # VOCAL REMOVER

    # sample, sr = librosa.load("Phlex_short.wav", mono=False)
    # l = Wave(sample[0, :])
    # r = Wave(sample[1, :])
    # SR = sr

    # out = 0.5 * (0.5*l - 0.5*r) + 0.5 * Lowpass(0.5*l + 0.5*r, cutoff=100)

    
    # OVERTONES

    out = 0.5 * Sin(200)
    for i in range(5):
        out += 0.1 * Sin(100*i)

    noise = Wave(np.random.rand(4000)-0.5)
    noise += out
    delay = Delay(None, delay=0.003)
    add = noise + delay
    delay.ins["a"] = 0.95 * add
    out = add

    dur = 10*SR

    t0 = time.time()
    out.eval(dur)
    elapsed = time.time() - t0

    print(elapsed)


    



    # TODO: declutter constants and TwoOps in the visualizer?

    


    # KARPLUS STRONG (without lowpass)

    # noise = Wave(np.random.rand(4000)-0.5)
    # delay = Delay(None, delay=0.003)
    # add = noise + delay
    # delay.ins["a"] = 0.95 * add
    # out = add

    # out = noise
    # for i in range(0, 100):
    #     out += 0.95 * Delay(out)

    
    g = drawgraph(out); g.render("gout", view=True)
    showsound(out, x2=150000)
    out.play(150000, live=False)


    # delayed = Delay(noise, delay=0.001)
    # plt.plot(noise.eval(5000))
    # plt.plot(delayed.eval(5000), alpha=0.5)
    # plt.show()


