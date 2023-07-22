from pynthlib import *
import pynthlib

def sines():
    out = 0.5 * Sin(200)
    for i in range(5):
        out += 0.1 * Sin(100*i)
    return out

def vocal():
    sample, sr = librosa.load("Phlex_short.wav", mono=False)
    l = Wave(sample[0, :])
    r = Wave(sample[1, :])
    #SR = sr
    out = 0.5 * (0.5*l - 0.5*r) + 0.5 * MovingAvg(0.5*l + 0.5*r, M=50)
    return out

def wind():
    noise = WhiteNoise()
    cutmod = Unipol(0.5*Sin(1/3) + 0.5*Sin(1/5)) * 0.1
    resmod = Unipol((0.5*Sin(1) + 0.5*Sin(0.8)) >> 0.3) * 0.4 + 0.1
    out = Lowpass(noise, cutmod, resmod)
    return out


def time_graphs(graphs, chunks, dur):

    print(len(graphs))
    res = np.zeros((len(graphs), len(chunks)))
    for i,g in enumerate(graphs):
        print(f"i = {i}")
        for j,ch in enumerate(chunks):
            print(f"ch = {ch}")
            t0 = time.time()
            pynthlib.CHUNK = ch
            g.eval(dur*SR)
            elapsed = time.time() - t0
            res[i, j] = elapsed
    return res

def eval():
    ratios = 0
    for i in range(3):
        dur = 3
        graphs = [sines(), vocal(), wind()]
        res = time_graphs(graphs, [10, 50, 100, 500], dur)
        ratios += res/dur
    print(ratios/3)

def eval_modules():

    control = Pulses(T=0.1)

    modules = {
        "sin": Sin(100),
        "saw": Saw(100),
        "noise": WhiteNoise(),
        "ramp": Ramp((0,1)),
        "trigwave": TriggWave(control, np.random.rand(10000)),
        "env": Envelope(control),
        "add": control + control,
        "mul": control * control,
        "delay": control >> 1,
        "lowpass": Lowpass(control),
        "modlowpass": Lowpass(control, control, control)
    }

    dur = 3
    res = time_graphs(modules.values(), [10, 50, 100], dur)/dur
    print(res.round(3))


if __name__ == "__main__":
    eval_modules()




