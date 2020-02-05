BLOB=10
NUM=1000000
SEQ_EXEC=seq
PTH_EXEC=parP
OMP_EXEC=parOmp

compil : lab1_omp.cpp lab1_pthread.cpp lab1_sequential.cpp lab1_omp.h lab1_pthread.h lab1_sequential.h lab1_io.c lab1_io.h main_omp.c main_pthread.c main_sequential.c
	g++ main_sequential.c lab1_sequential.cpp lab1_io.c -o $(SEQ_EXEC) -lgomp
	g++ main_pthread.c lab1_pthread.cpp lab1_io.c -o $(PTH_EXEC) -lpthread -lgomp
	g++ main_omp.c lab1_omp.cpp lab1_io.c -o $(OMP_EXEC) -lgomp -fopenmp

getstats : $(SEQ_EXEC) $(PTH_EXEC) $(OMP_EXEC)
	./$(SEQ_EXEC) $(BLOB) dataset_$(NUM)_$(BLOB).txt $(SEQ_EXEC)_a.txt $(SEQ_EXEC)_b.txt
	
	echo "pThread"
	./$(PTH_EXEC) $(BLOB) 2 dataset_$(NUM)_$(BLOB).txt pthread2_a.txt pthread2_b.txt
	./$(PTH_EXEC) $(BLOB) 3 dataset_$(NUM)_$(BLOB).txt pthread3_a.txt pthread3_b.txt
	./$(PTH_EXEC) $(BLOB) 5 dataset_$(NUM)_$(BLOB).txt pthread5_a.txt pthread5_b.txt
	./$(PTH_EXEC) $(BLOB) 6 dataset_$(NUM)_$(BLOB).txt pthread6_a.txt pthread6_b.txt
	./$(PTH_EXEC) $(BLOB) 10 dataset_$(NUM)_$(BLOB).txt pthread10_a.txt pthread10_b.txt
	./$(PTH_EXEC) $(BLOB) 16 dataset_$(NUM)_$(BLOB).txt pthread16_a.txt pthread16_b.txt
	./$(PTH_EXEC) $(BLOB) 32 dataset_$(NUM)_$(BLOB).txt pthread32_a.txt pthread32_b.txt
	
	echo "openMP"
	./$(OMP_EXEC) $(BLOB) 2 dataset_$(NUM)_$(BLOB).txt omp2_a.txt omp2_b.txt
	./$(OMP_EXEC) $(BLOB) 3 dataset_$(NUM)_$(BLOB).txt omp3_a.txt omp3_b.txt
	./$(OMP_EXEC) $(BLOB) 5 dataset_$(NUM)_$(BLOB).txt omp5_a.txt omp5_b.txt
	./$(OMP_EXEC) $(BLOB) 6 dataset_$(NUM)_$(BLOB).txt omp6_a.txt omp6_b.txt
	./$(OMP_EXEC) $(BLOB) 10 dataset_$(NUM)_$(BLOB).txt omp10_a.txt omp10_b.txt
	./$(OMP_EXEC) $(BLOB) 16 dataset_$(NUM)_$(BLOB).txt omp16_a.txt omp16_b.txt
	./$(OMP_EXEC) $(BLOB) 32 dataset_$(NUM)_$(BLOB).txt omp32_a.txt omp32_b.txt

.PHONY: clean

clean: 
	rm $(SEQ_EXEC) $(PTH_EXEC) $(OMP_EXEC)
	rm pthread*.txt omp*.txt seq*.txt