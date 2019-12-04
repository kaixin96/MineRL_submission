# Code for MineRL competition submission

This repo contains code for our team(CraftRL)'s [MineRL competition](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition) submission.

### Dependencies

* Python 3.6 +
* PyTorch 1.1.0
* NumPy
* gym 0.12.5
* minerl 0.2.9
* rlpyt

### Hacking `rlpyt`

In order to run parallel envs, you need to hack `rlpyt`. If parallel processes start at the same time, `minerl` will assign the same port to them, which will crash the environment. A workaround is to add `time.sleep(60)` after `w.start()` in `ParallelSamplerBase.initialize()` in `rlpyt/samplers/parallel/base.py`.