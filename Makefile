CFLAGS=-fPIC -std=c++11 -Wall -O3 -g -fopenmp
SO=_sib$(shell python3-config --extension-suffix)
LIN=
PYINC=$(shell python3 -m pybind11 --includes)


ifneq (${CXX},"")
    INC_GEN=-I${CONDA_PREFIX}/include
    LINK=-lgomp -lm -L${CONDA_PREFIX}/lib
else
    CXX=g++
    INC_GEN=
    LINK=-lgomp -lm
endif

all: sib ${SO}

bp.o: bp.cpp bp.h cavity.h
	${CXX} ${CFLAGS} ${INC_GEN} -c bp.cpp -o $@ 
sib: bp.o sib.cpp
	${CXX} ${CFLAGS} bp.o sib.cpp ${LINK} ${INC_GEN} -o $@
${SO}: bp.o pysib.cpp
	${CXX}  -shared ${CFLAGS} ${PYINC} ${INC_GEN} ${LINK} pysib.cpp bp.o -o $@

clean:
	rm -f sib ${SO} *.o
