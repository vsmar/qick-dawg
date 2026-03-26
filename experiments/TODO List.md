# Ideas to implement

1. Have all key plotting addressable via a function to facilitate postprocessing
2. Improve unified plotting python file (with post processing support)
3. Code seperation (currently scripts are mixed in with each other)
4. Laser script with powershell alias calls
5. powershell aliases for running the experiments (ideal would be something like rwreopt -reps=1000000 -name=CPMG)
6. Look into modifying reoptimization to run without manual instruction
7. Secure reoptimization (currently the file is untracked, next upload could wipe it. Should it be part of vqlmutils?)
8. Migrate the data elsewhere (currently it gets saved locally(fine), but also isn't ignored by the git repo, not so good)

REOPTIMIZATION LOOP:
4. Create a shared reopt structure that can either be passed a an experiment or called from an experiment (implement experiments as classes?)
5. Have reopt automatically determine how to divy up reps
6. Improve statistic gathering (averages and SD of steady state in runs, maybe SD within a run) - analysis of the chunked data

Files are currently spread between:
src/nvtestsuite/ (mixed with one non fine res script i wrote for pulsed odmr)
experiments/
and jupyter notebooks (mixed with all other notebooks)
nvpulsing/nvconfiguration.py (made edits which are critical to current fine timing implementation)