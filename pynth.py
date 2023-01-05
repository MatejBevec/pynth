import os
import sys
import math
import numbers
import inspect
import time
from threading import Thread

import numpy as np
import scipy as sp 
import librosa
import sounddevice as sd
import matplotlib.animation as animation
#matplotlib.use("webagg")
import matplotlib.pyplot as plt
from graphviz import Digraph

import keyboard

SR = 48000 # TEMP 
CHUNK = 200 # TIME CHUNK LEN IN SAMPLES
MAXDEPTH = 30
gtime = 0 # CURRENT CHUNK ON THE CLOCK - TEMP
globnodes = []
globscopes = []
SCOPERATE = 8
anim = None # necessary matplotlib thing


# TODO: MODULATING PARAMETERS
    # TODO: either modules like Delay, Oscillators etc. take a "control" input
    # TODO: or there is a wrapped Modulate, that can change any parameters of the wrapped module

# TODO: INPUTS - how should they be passed in and modified later
    # TODO: modify via controlled method, so everything is recomputed properly
    # TODO: Compose - control input setting - the composed module needs to be updated on input changes

# TODO: consider using factory methods like sin(), then saw() can return a special Triangle

# TODO: auto normalize signals

# TODO: if we assume tiny chunks, some modules can probably be simplified

# TODO: stateful modules need to have some sort of _reset function that gets
# on every new eval() call

# TODO: how to handle multiple outputs on modules like sequencer/piano roll



#   --------- HELPER FUNCTIONS ----------

def toposort(out):
    topo = []
    visited = set()
    def _build(v):
        if v is not None and v not in visited:
                visited.add(v)
                for child in v.ins.values():
                    _build(child)
                topo.append(v)
    _build(out)
    return topo

def _livescopes(dur):

    print("Starting scopes...")

    fig = plt.figure(figsize=(12, 6))
    fig.tight_layout()
    nscopes = len(globscopes)
    axes = [fig.add_subplot(nscopes, 1, i+1) for i in range(nscopes)]
    for i,ax in enumerate(axes):
        ax.set_xlim(-globscopes[i].window, 0)
        ax.set_ylabel(f"{globscopes[i].ins['a']} -> {globscopes[i]}")
    initx = np.linspace(-1/SCOPERATE, 0, int(SR/SCOPERATE))
    inity = np.zeros_like(initx)
    plots = [ax.plot(initx, inity)[0] for ax in axes]

    frames = int((dur/SR)*SCOPERATE)
    interval = interval=int(1000/SCOPERATE)

    def animate(at):
        for i in range(len(plots)):
            x, y = globscopes[i].getlast()
            plots[i].set_data(x, y)
            axes[i].set_ylim(min(np.min(y),-1) - 0.05, max(np.max(y), 1) + 0.05)

        if at == frames-1:
            plt.close()

    global anim
    anim = animation.FuncAnimation(fig, animate, repeat=False, 
                                frames=frames, interval=interval)
    plt.show()

def _showscopes():
    nscopes = len(globscopes)
    for i in range(nscopes):
        plt.subplot(nscopes, 1, i+1)
        scope = globscopes[i]
        x, y = globscopes[i].getdata(None)
        plt.plot(x, y)
    plt.tight_layout(pad=1.0)
    plt.show()


def get_crossings(a, lastprev=0, thr=0.5):
    a = np.insert(a, 0, lastprev)
    mask = (a > thr)
    shifted = np.zeros(len(mask)).astype(bool)
    shifted[1:] = mask[:-1]
    imp = np.logical_and(mask, np.logical_not(shifted))
    imp = imp[1:]
    loc = np.nonzero(imp)[0]
    return loc

def clamp_signal(a, range=[0.001, 0.999]):
    print(a)
    a = np.array(a)
    inmin = np.min(a)
    inrange = np.max(a) - np.min(a)
    print(inmin, inrange)
    a -= inmin; a /= inrange
    print(a)
    outrange = range[1] - range[0]
    a *= outrange; a += range[0]
    print(a)
    return a

def to_unipolar(a):
    """From [-1, 1] to [0, 1]"""
    a = np.clip(np.array(a), -1, 1)
    return (a + 1) / 2

def to_bipolar(a):
    """From [0, 1] to [-1, 1]"""
    a = np.clip(np.array(a), 0, 1)
    return a * 2 - 1



#   --------- MODULE CLASSES ----------

class Module():
    """Base class"""

    def __init__(self):
        # self.ins = {}
        # undefined inputs are always treated as zero signals
        if not hasattr(self, "ins"): self.ins = {}
        self.indata = {inn: np.zeros(CHUNK) for inn in self.ins}
        self.outdata = np.zeros(CHUNK)
        self._mem = {}
        self._ready = False

        self.i = len(globnodes)
        globnodes.append(self)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.i})"

    def _compute(self, t, d=0):
        """Compute the next chunk.
        
        Args:
            t: time (samples) at the beginning of this chunk
            d: depth, i.e. how many times this has been called at this t, to stop infinite loops
        """
        pass

    def proc(self, t):
        """Process one chunk for all upstream modules"""
        topo = toposort(self)

        for node in topo:
            node._proc(t, 0)

    def _proc(self, t, d=0):
        """Process one chunk for this module"""

        for inn in self.ins:
            if self.ins[inn] is not None:
                self.indata[inn] = self.ins[inn].outdata
        if d > MAXDEPTH:
            return
        chunk = self._compute(t, d)
        self.outdata = chunk

    def eval(self, dur):
        """Process [0, dur/CHUNK] chunks for all modules and concat outputs"""

        data = []
        for t, smp in enumerate(range(0, dur, CHUNK)):
            self.proc(t)
            data.append(self.outdata)
        return np.concatenate(data)[:dur]


    def play(self, dur, live=False, callback=None, scopes=True):
        """Play the output from this node for dur samples"""

        scopes = scopes and len(globscopes) > 0

        if not live:
            data = self.eval(dur)
            if scopes: _showscopes()
            sd.play(data, SR)
            sd.wait()
            return

        # TODO: decide if we can have global scopes or do we need concurrent comp. graph option
        
        if scopes:
            thread = Thread(target=_livescopes, args=(dur,))
            thread.start()

        global gtime; gtime = 0
        def _callback(outdata, frames, t, status):
            global gtime
            ch = self.proc(gtime)
            outdata[:, 0] = self.outdata
            outdata[:, 1] = self.outdata
            if callback is not None: callback(gtime*CHUNK/SR)
            gtime += 1
            
        with sd.OutputStream(samplerate=SR, blocksize=CHUNK, callback=_callback):
            sd.sleep(int(dur/SR * 1000))

        if scopes:
            thread.join()


    # --- DUNDER OPERATIONS ---

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

        # TODO: converting from chunks to seconds to samples is numerically awkward

    def _getdata(self, t, d=0):
        reqst, end = int(round(t*SR)), int(round(t*SR+CHUNK))
        # reqst, end = (t)*CHUNK, (t+1)*CHUNK
        st, neg = max(reqst, 0), min(reqst, 0)
        chunk = None
        if isinstance(self.data, numbers.Number):   # data can be a constant
            chunk = np.full((end - st,), self.data)
        if isinstance(self.data, np.ndarray):       # data can be an array of samples
            l = len(self.data)
            chunk = self.data[min(st, l) : min(end, l)]
            chunk = np.pad(chunk, (0, end-st - len(chunk)), constant_values=self._pad)
        if callable(self.data):                     # data can be a function over time
            chunk = self.data(np.linspace(st/SR, end/SR, end-st))
        chunk = np.pad(chunk, (-neg, 0))
        assert len(chunk) == CHUNK
        return chunk

    def _compute(self, t, d=0):
        tsec = t*(CHUNK/SR)
        return self._getdata(tsec, d)



# TODO: a better user intf would be to have a method like
# or trigger(Sin(fe), control) or trigger(Sin(fe))(control)

# TODO: this behavior should be generalized if small chunks are assumed
# all the modules need is a local time

class TriggWave(Wave):
    """A signal generator that resets at every input front"""

    def __init__(self, incontrol, data, pad=0):
        self.ins = {"control": incontrol}
        self.lastt = None
        self.lastprev = 1
        super().__init__(data)

    def _compute(self, t, d=0):
        control = self.indata["control"]
        loc = get_crossings(control, self.lastprev)
        out = np.zeros(CHUNK)
        tsec = t*CHUNK / SR
        tarray = np.arange(CHUNK) / SR + tsec
        self.lastprev = control[-1]

        if self.lastt is not None:
            if self.lastt > tsec: self.lastt = tsec
            elapsed = tsec - self.lastt
            out = self._getdata(elapsed, d)
        for i in loc:
            self.lastt = tarray[i]
            wave = self._getdata(0, d)
            out[i:] = wave[:len(wave)-i]

        return out


class Input(Wave):
    """Generate output level with function call"""

    # TODO: Multithreaded Async instead of callback?
    # TODO: if you can have more than one keypress in a chunk
    # then other module need to stay as they are
    # if we limit it to one per chunk, it can be simplified

    def __init__(self):
        super().__init__(0)

    def set(self, value):
        assert isinstance(value, numbers.Number)
        self.data = value

        
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



class Envelope(TriggWave):
    """Output an envelope signal at every input front"""

    # TODO: ADSR not just AD

    def __init__(self, ina, durs=(4000, 2000, 0, 10000), suslvl=0.7, thr=0.5):
        self.durs = durs
        self.suslvl = suslvl
        self.thr = thr
        # sus is the sustain duration if key is released immediately
        att, dec, sus, rel, = durs 

        attack = np.linspace(0, 1, att)
        decay = np.linspace(1, 0, dec)**2 * (1-suslvl) + suslvl
        #sustain = np.ones(20000) * suslvl
        release = np.linspace(1, 0, rel)**2 * suslvl

        data = np.concatenate([np.linspace(0, 1, att), np.linspace(1, 0, rel)**2])
        print(data.shape)
        super().__init__(ina, data)


class Sequencer(Wave):
    """Output next bit from a sequence at every input front"""

    def __init__(self, ina, sequence):
        self.ins = {"a": ina}
        if isinstance(sequence, list):
            sequence = np.array(sequence)
        self.sequence = sequence
        self.state = 0
        self.lastprev = 1
        super().__init__(None)

    def data(self, t):
        a = self.indata["a"]
        out = np.zeros(len(t))
        loc = get_crossings(a, lastprev=self.lastprev)
        self.lastprev = a[-1]
        prev = 0
        for i in loc:
            out[prev:i] = float(self.sequence[self.state])
            out[i-1] = 0.0
            self.state += 1
            if self.state == len(self.sequence): self.state = 0
            prev = i
        out[prev:] = float(self.sequence[self.state])
        return out


#   --------- BASIC OSCILLATORS ----------

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
        # TODO: this can be cleaned up
        self.data = lambda t: 1 + 2 * ((t%p <= rtime) * ((t%p)  / rtime - 1) - (t%p > rtime ) * (((t%p) - rtime) / (p - rtime)))

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

class WhiteNoise(Wave):

    def __init__(self):
        super().__init__(None)
        self.data = lambda t: np.random.rand(len(t))*2 - 1



#   --------- ARITHMETIC AND LOGIC OPERATIONS ----------

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



class Unipol(Module):
    """Transform bipolar [-1, 1] signal to unipolar [0, 1]"""

    def __init__(self, ina=None):
        self.ins = {"a": ina}
        super().__init__()
    
    def _compute(self, t, d=0):
        return to_unipolar(self.indata["a"])

class Bipol(Module):
    """Transform unipolar [0, 1] signal to bipolar [-1, 1]"""

    def __init__(self, ina=None):
        self.ins = {"a": ina}
        super().__init__()
    
    def _compute(self, t, d=0):
        return to_bipolar(self.indata["a"])



class Delay(Module):
    """Output a delayed input signal"""

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



class FreqFilter(Module):
    """Filter with given frequency response"""

    # TBA: basically an FFT implementation of EQ

    def __init__(self):
        # 1. get and store freq_response
        pass

    def freq_response():
        # standardize to 2048 bins from 20Hz to 20000Hz?
        pass

    def _compute(self, t, d=0):
        # 1. fetch the correct window
        # 2. zero pad
        # 3. FFT
        # 4. multiply X by freq_response
        # 5. IFFT
        # 6. sum with prev buffer
        # 7. store next buffer
        pass



# TODO: generalize filters and modulation

class Filter(Module):
    """Base class for LTI filters"""

    def __init__(self):
        # A filter should set self.a and self.b at init
        # and expose _getparam which returns new a, b according to control signal
        print(self.a)
        self.zi = sp.signal.lfilter_zi(self.b, self.a)
        super().__init__()

    def _setparam(self):
        """Reconstruct filter with updated params. Do not override."""

        if self._getparam is None: return
        param = self._getparam()
        if param is None: return
        self.b, self.a = param

        znew = sp.signal.lfilter_zi(self.b, self.a)
        if len(self.zi) > len(znew):
            self.zi = self.zi[-len(znew):]
        else:
            znew[:len(self.zi)] = self.zi
            self.zi = znew

    def _getparam(self):
        """Computes params according to control inputs and returns b, a.
            Override this method."""
        pass
        
    def _compute(self, t, d=0):
        self._setparam()
        x = self.indata["a"]
        y, zf = sp.signal.lfilter(self.b, self.a, x, zi=self.zi)
        self.zi = zf
        return y

class Butter(Filter):
    """Butterworth filter from provided params"""

    # BUG: impulse artifact

    def __init__(self, ina=None, incontrol=None, crit=0.3, btype="lowpass", fs=SR):
        self.ins = {"a": ina, "control": incontrol}
        self.crit = crit
        self.btype = btype
        self.b, self.a = sp.signal.butter(2, self.crit, self.btype, analog=False)
        super().__init__()

    def _getparam(self):
        if self.ins["control"] is None: return
        # TODO: cap at [0, 1]
        crit = self.indata["control"][0]
        assert crit >= 0 and crit <= 1
        #print(crit)
        b, a = sp.signal.butter(2, crit*0.99+0.005, self.btype, analog=False)
        return b, a

class MovingAvg(Filter):
    """Moving average lowpass filter"""
    
    def __init__(self, ina=None, incontrol=None, M=50):
        self.ins = {"a": ina, "control": incontrol}
        self.M = M
        self.b, self.a = np.ones(M)/M, [1]
        super().__init__()

    def _getparam(self):
        if self.ins["control"] is None: return
        val = self.indata["control"][0]
        assert val  >= 0 and val <= 1
        M = int((1-val)*self.M)
        b, a = np.ones(M)/M, [1, 0]
        return b, a

class EQ(Filter):
    """Window-function-based custom EQ"""

    def __init__(self, ina=None, freq=[20, 20000], gain=[1, 1]):
        self.ins = {"a": ina}
        self.b, self.a = sp.signal.firwin2(100, freq, gain)
        super().__init__()

class Lowpass(Filter):
    """Lowpass with modulated cutoff and resonance"""

    def __init__(self, ina=None, incontrol=None, inres=None, cutoff=0.5, res=0.2):
        self.ins = {"a": ina, "control": incontrol, "res": inres}
        self.cutoff = cutoff
        self.res = res
        self.b, self.a = self._getfilt(cutoff, res)
        super().__init__()

    def _getfilt(self, cutoff, res):
        cband = 0.2 # width of cutoff transition
        rband = 0.05 # width of transition to resonance peak
        cutoff = max(min(0.99, cutoff), 0.01)
        freq = [0, max(cutoff-rband, 0.01), cutoff, min(cutoff+cband, 0.99), 1]
        gain = [1-res, 1-res, 1, 0, 0]
        return sp.signal.firwin2(100, freq, gain), [1, 0]

    def _getparam(self):
        inc = self.ins["control"] is not None
        inr = self.ins["res"] is not None
        if not (inc or inr): return
        if inc: self.cutoff = self.indata["control"][0]
        if inr: self.res = self.indata["res"][0]
        return self._getfilt(self.cutoff, self.res)



#   --------- COMPOSED MODULES ----------

# TODO: cleaner
# TODO: compose() should accept parameters too

def compose(factory, label=None):
    """Create a composed module from a lambda expression"""

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
    """A module that wraps a subgraph"""

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



#   --------- VISUALIZATION ----------
    
def showsound(module, t1=0, t2=30000, sec=False):
    """Evaluate module and plot output sound"""

    if sec:
        t1, t2 = int(t1*SR), int(t2*SR)
    data = module.eval(t2)
    x = np.linspace(0, t2, t2)
    if sec:
        x /= SR
        t1, t2 = t1/SR, t2/SR
    plt.plot(x, data)
    plt.xlim((t1, t2))
    plt.show()


# TODO: standardize labels/names
# TODO: now "sinks" that are not called dont get put into the comp. graph, should we change this?

class Scope(Module):
    """Visualize input signal at play time"""

    def __init__(self, ina, window=0.25, dist=0.5):
        self.ins = {"a": ina}
        self.window = window
        globscopes.append(self)
        self.history = np.zeros(int(SR*self.window))
        self.dist = dist
        self.distsamp = int(SR*self.window)
        super().__init__()
        
    def _compute(self, t, d=0):
        data = self.indata["a"]
        self.history = np.concatenate([self.history, data])[-self.distsamp:]
        return data

    def getdata(self, dur):
        dursamp = int(dur*SR) if dur else len(self.history)
        y = self.history[-dursamp:]
        x = np.linspace(-dursamp/SR, 0, dursamp)
        return x, y
    
    def getlast(self):
        return self.getdata(self.window)

    def show(self, dur, ax=None):
        x, y = self.getdata(dur)
        if ax is None: ax = plt
        ax.plot(x, y)
        ax.title(f"{self} showing {self.ins['a']}")

    
        
def trace(root):
    root._isroot = True
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.ins.values():
                if child is not None:
                    edges.add((child, v))
                    build(child)
    build(root)
    return nodes, edges

def drawgraph(root, format='svg', rankdir='LR', render=True):
    """Draw the computation graph given the root (output) node.

    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        isroot = hasattr(n, "_isroot") and n._isroot
        color = 'black'; fontcolor = 'black'
        if isinstance(n, Wave): color = 'blue'
        if isroot: color = 'red'
        label = n.__class__.__name__

        # TODO: put this logic in the classes

        if isinstance(n, Add): label = '+'
        if isinstance(n, Mul): label = "*"
        if isinstance(n, Wave) and isinstance(n.data, numbers.Number):
            if not isinstance(n, Input):
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
        if isinstance(n1, Wave) and isinstance(n1.data, numbers.Number):
            if not isinstance(n1, Input):
                color = 'gainsboro'
        dot.edge(str(id(n1)), str(id(n2)), color=color) # + n2._op
    
    dot.render("gout", view=True)
    return dot



if __name__ == "__main__":

    # OVERTONES

    # out = 0.5 * Sin(200)
    # for i in range(5):
    #     out += 0.1 * Sin(100*i)

    # drawgraph(out)
    # showsound(out, t2=0.1, sec=True)
    # out.play(5*SR)


    # SAMPLES, VOCAL REMOVER

    sample, sr = librosa.load("Phlex_short.wav", mono=False)
    l = Wave(sample[0, :])
    r = Wave(sample[1, :])
    SR = sr
    
    out = 0.5 * (0.5*l - 0.5*r) + 0.5 * MovingAvg(0.5*l + 0.5*r, M=50)
    drawgraph(out)
    showsound(out, t2=5, sec=True)
    out.play(10*SR)

    
    # KARPLUS, FREEZING SAMPLES

    # noise = Wave(np.random.rand(4000)-0.5)
    # delay = Delay(None, delay=0.005)
    # add = noise + delay
    # delay.ins["a"] = 0.95 * MovingAvg(add, M=10)
    # out = add
    # frozen = Wave(add.eval(2*SR))
    # frozen.play(2*SR)

    # drawgraph(out); showsound(frozen, t2=2, sec=True)
    #out.play(2*SR)


    # SEQUENCER

    # clock = Square(4)
    # seq = Sequencer(clock, [1, 0, 1, 0, 1, 1, 1, 0]) >> 0.01
    # env = Envelope(seq, durs=(1000, 0, 0, 6000))

    # out = env * Saw(200) * Sin(50)


    # ENVELOPES, USER INPUT, SCOPES

    # control = Input()
    # env = Envelope(Scope(control), (2000, 0, 0, 8000))
    # out = Scope(env)
    # out = Scope(out * Saw(100))

    # val = 0
    # def loop(t):
    #     if keyboard.is_pressed('l'):
    #         control.set(1)
    #     else:
    #         control.set(0)

    # out.play(30*SR, live=True, callback=loop)


    # LOWPASS SWIPE

    # sample, sr = librosa.load("house.mp3", mono=True)
    # SR = sr
    # music = Wave(sample)    
    # control = Scope(Sin(1/5)*0.5+0.5)
    # out = Lowpass(music, control, res=0.4)
    # out = Scope(out)
    # out.play(20*SR, live=True, callback=None)


    # PROCEDURAL WIND

    # noise = WhiteNoise()
    # cutmod = Unipol(0.5*Sin(1/3) + 0.5*Sin(1/5)) * 0.1
    # resmod = Unipol((0.5*Sin(1) + 0.5*Sin(0.8)) >> 0.3) * 0.4 + 0.1
    # out = Scope(Lowpass(noise, cutmod, resmod))
    # drawgraph(out)
    # showsound(out, t2=0.5, sec=True)
    #out.play(30*SR, live=False)

    
    # THIS IS ACID MAAAN

    # clock = Pulses(T=1/8)
    # clock = Sequencer(clock, [0,1,0,1,0,1,1,1])
    # env = Scope(Envelope(clock, durs=(1000, 0, 0, 5000)))
    # voice = Saw(80)
    # out = env * Lowpass(voice, env, res=0.6)
    # out = Scope(out)
    # t0 = time.time()
    # drawgraph(out)
    # showsound(out, t2=3, sec=True)
    # out.play(20*SR, live=False)
    # print(time.time() - t0)


    # TODO: should samples or seconds be the standard unit of time ?
    # TODO: should the interface instead be Module(params)(inputs)
    # TODO: cap outputs at [-1, 1] ?
    # TODO: kill all threads if main crashes

