FC = gfortran

linden.x:	linden.o
		$(FC) -o $@ linden.o

linden.o:	linden.f
		$(FC) -c linden.f

clean:
		rm -f *.o linden.x
