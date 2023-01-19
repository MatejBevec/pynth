![cover](figs/logo.png)

Intuitive sound synthesis in python.



## Installation

```
git clone https://github.com/MatejBevec/pynth
pip install ./pynth
```

## Usage

Below are some usage examples. 
For more info, feel free to read `DESCRIPTION.pdf` and check out `examples.py`.

### Overtones
```python
from pynth import *
out = 0.5 * Sin(200)
for i in range(5):
out += 0.1 * Sin(100*i)
drawgraph(out)
showsound(out, t2=0.1)
out.play(5*SR)
```
<img src="figs/overtones.png" width=900>

https://user-images.githubusercontent.com/40042371/213463015-5cdd5f04-aa9d-4fe6-bfb3-5fa47394b6f1.mp4

### Vocal remover
```python
sample, sr = librosa.load("file.wav", mono=False)
l = Wave(sample[0, :])
r = Wave(sample[1, :])
SR = sr
out = 0.5 * (0.5*l - 0.5*r)
out += 0.5 * MovingAvg(0.5*l + 0.5*r, M=50)
```

<img src="figs/vocal.png" width=900>

https://user-images.githubusercontent.com/40042371/213463075-75020d90-9f85-4ac3-9728-504b07fb9995.mp4

### Procedural wind
```python
noise = WhiteNoise()
cutmod = Unipol(.5*Sin(1/3) + .5*Sin(1/5)) * 0.1
resmod = Unipol((.5*Sin(1) + .5*Sin(.8)) >> 0.3)
resmod = resmod * 0.4 + 0.1
out = Scope(Lowpass(noise, cutmod, resmod))
```

<img src="figs/wind.png" width=900>

https://user-images.githubusercontent.com/40042371/213463145-76281843-c404-4aa1-a73d-95d23d691cea.mp4

### Karplus-Strong string emulator
```python
noise = Wave(np.random.rand(4000)-0.5)
delay = Delay(None, delay=0.005)
add = noise + delay
delay.ins["a"] = 0.95 * MovingAvg(add, M=10)
out = add

frozen = Wave(add.eval(2*SR))
```

<img src="figs/karplus.png" width=900>

https://user-images.githubusercontent.com/40042371/213463212-e894d9f7-7ee3-4b75-9c26-2594c877b96c.mp4

### User input, envelopes and scopes
```python
import keyboard
control = Input()
env = Envelope(Scope(control), (2000, 0, 0, 8000))
out = Scope(env)
out = Scope(out * Saw(100))
val = 0
def loop(t):
if keyboard.is_pressed(’l’):
control.set(1)
else:
control.set(0)
out.play(30*SR, live=True, callback=loop)
```

<img src="figs/input.png" width=900>
