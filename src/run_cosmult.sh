#!/bin/bash

#cosmult
#src/main.py data/glove.6B.50d.txt data/google/ data/out/50d/google_cosmult/ -d cosmult
#src/main.py data/glove.6B.100d.txt data/google/ data/out/100d/google_cosmult/ -d cosmult
src/main.py data/glove.6B.200d.txt data/google/ data/out/200d/google_cosmult/ -d cosmult
#src/main.py data/glove.6B.50d.txt data/msr/ data/out/50d/msr_cosmult/ -d cosmult
#src/main.py data/glove.6B.100d.txt data/msr/ data/out/100d/msr_cosmult/ -d cosmult
src/main.py data/glove.6B.200d.txt data/msr/ data/out/200d/msr_cosmult/ -d cosmult
