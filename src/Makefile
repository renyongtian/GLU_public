# Makefile
DEBUG = 0


LU_INC = -Ipreprocess -I../include -Inicslu/include

PREPROCESS_LIBS = preprocess/preprocess.o nicslu/lib/nicslu.a nicslu/util/nicslu_util.a

NVCC = nvcc
CPP = g++
EXE = lu_cmd
# 调试模式配置
DEBUG_CFLAGS = -O0 -g -m64 -Wall -std=c++11 $(LU_INC) -DGLU_DEBUG=${DEBUG}
DEBUG_NVCCFLAGS = -O0 -g -G -arch=sm_80 -std=c++11 -Xcompiler $(LU_INC)

# 默认使用发布模式（-O3），通过命令切换为调试模式
ifeq ($(DEBUG), 1)
    CFLAGS = $(DEBUG_CFLAGS)
    NVCCFLAGS = $(DEBUG_NVCCFLAGS)
else
    CFLAGS = -O3 -m64 -Wall -std=c++11 $(LU_INC) -DGLU_DEBUG=${DEBUG}
    NVCCFLAGS = -O3 -lineinfo -arch=sm_80 -std=c++11 -Xcompiler $(LU_INC)
endif
CFLAGS = -O3 -m64 -Wall -std=c++11 $(LU_INC) -DGLU_DEBUG=${DEBUG}
NVCCFLAGS = -O3 -lineinfo -arch=sm_80 -std=c++11 -Xcompiler $(LU_INC)

MAIN: $(EXE)

lu_cmd: Timer.o numeric.o symbolic.o lu_cmd.o $(PREPROCESS_LIBS)
	$(NVCC) $^ -o $@

Timer.o: Timer.cpp
	$(CPP) $(CFLAGS) -c $< -o $@

numeric.o: numeric.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

symbolic.o: symbolic.cc
	$(CPP) $(CFLAGS) -c $< -o $@

lu_cmd.o: lu_cmd.cpp
	$(CPP) $(CFLAGS) -c $< -o $@

preprocess/preprocess.o:
	make -C preprocess/

nicslu/lib/nicslu.a:
	make -C nicslu/

nicslu/lib/nicslu_util.a:
	make -C nicslu/

clean:
	rm -f *.o *~ *.so $(EXE)
	make -C preprocess/ clean

distclean:
	rm -f *.o *~ *.so $(EXE)
	make -C nicslu/ clean
	make -C preprocess/ clean


