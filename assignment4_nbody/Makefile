nbody: main.cu gpu.cu common.h
	nvcc --gpu-architecture compute_86 -O2 main.cu gpu.cu -o nbody


serial: serial.cpp
	g++ -O2 serial.cpp -o serial

clean:
	rm -f nbody serial

submit:
	zip $(USER)_assignment4.zip  gpu.cu
	cp $(USER)_assignment4.zip /mnt/data1/submissions/assignment4/
