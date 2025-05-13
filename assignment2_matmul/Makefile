NVCC ?= /usr/local/cuda-12.6/bin/nvcc

matmul: benchmark.cpp template.cu
	$(NVCC) $^ -lcublas -o $@

clean:
	rm -f a.out matmul

submit:
	zip $(USER)_assignment2.zip Makefile *.cu *.cuh *.hpp *.cpp
	cp $(USER)_assignment2.zip /mnt/data1/submissions/assignment2/