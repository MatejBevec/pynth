import keyboard

import librosa
import soundfile
import numpy as np
from pynthlib import pynth as pt

if __name__ == "__main__":

    # OVERTONES

    out = 0.5 * pt.Sin(200)
    for i in range(5):
        out += 0.1 * pt.Sin(100*i)

    pt.drawgraph(out)
    pt.showsound(out, t2=0.1, sec=True)
    out.play(5*pt.SR)


    # SAMPLES, VOCAL REMOVER
    input("Next example - vocal remover (press any key):")

    sample, sr = librosa.load("pop.wav", mono=False)
    l = pt.Wave(sample[0, :])
    r = pt.Wave(sample[1, :])
    pt.SR = sr
    
    out = 0.5 * (0.5*l - 0.5*r) + 0.5 * pt.MovingAvg(0.5*l + 0.5*r, M=50)
    pt.drawgraph(out)
    pt.showsound(out, t2=5, sec=True)
    out.play(10*pt.SR)
    soundfile.write("vocal-remover.wav", out.eval(5*pt.SR), pt.SR)

    
    # KARPLUS, FREEZING SAMPLES
    input("Next example - Karplus Strong (press any key):")

    noise = pt.Wave(np.random.rand(4000)-0.5)
    delay = pt.Delay(None, delay=0.005)
    add = noise + delay
    delay.ins["a"] = 0.95 * pt.MovingAvg(add, M=10)
    out = add
    frozen = pt.Wave(add.eval(2*pt.SR))
    frozen.play(2*pt.SR)

    pt.drawgraph(out)
    pt.showsound(frozen, t2=2, sec=True)
    out.play(2*pt.SR)


    # ENVELOPES, USER INPUT, SCOPES
    input("Next example - user input, envelopes and scopes (press any key):")

    control = pt.Input()
    env = pt.Envelope(pt.Scope(control), (2000, 0, 0, 8000))
    out = pt.Scope(env)
    out = pt.Scope(out * pt.Saw(100))

    val = 0
    def loop(t):
        if keyboard.is_pressed('l'):
            control.set(1)
        else:
            control.set(0)

    out.play(30*pt.SR, live=True, callback=loop)


    # LOWPASS SWIPE
    input("Next example - lowpass swipe (press any key):")

    sample, sr = librosa.load("house.mp3", mono=True)
    pt.SR = sr
    music = pt.Wave(sample)    
    control = pt.Scope(pt.Sin(1/5)*0.5+0.5)
    out = pt.Lowpass(music, control, res=0.4)
    out = pt.Scope(out)
    out.play(20*pt.SR, live=True, callback=None)


    # PROCEDURAL WIND
    input("Next example - precedural wind generator (press any key):")

    noise = pt.WhiteNoise()
    cutmod = pt.Unipol(0.5*pt.Sin(1/3) + 0.5*pt.Sin(1/5)) * 0.08
    resmod = pt.Unipol((0.5*pt.Sin(1) + 0.5*pt.Sin(0.8)) >> 0.3) * 0.4 + 0.2
    out = pt.Scope(pt.Lowpass(noise, cutmod, resmod))

    pt.drawgraph(out)
    pt.showsound(out, t2=0.5, sec=True)
    #out.play(30*pt.SR, live=False)
    soundfile.write("wind.wav", out.eval(20*pt.SR), pt.SR)

    
    # THIS IS ACID MAAAN
    input("Next example - acid baseline (press any key):")

    clock = pt.Pulses(T=1/8)
    clock = pt.Sequencer(clock, [0,1,0,1,0,1,1,1])
    env = pt.Scope(pt.Envelope(clock, durs=(1000, 0, 0, 5000)))
    voice = pt.Saw(80)
    out = env * pt.Lowpass(voice, env, res=0.6)
    out = pt.Scope(out)

    pt.drawgraph(out)
    pt.showsound(out, t2=3, sec=True)
    out.play(20*pt.SR, live=False)

