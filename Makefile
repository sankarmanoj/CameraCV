CC=nvcc
CFLAGS= -ccbin=g++-6   `pkg-config --cflags  opencv` -L.
LDFLAGS= `pkg-config --libs opencv` -ldarknet -lcudnn -lcublas -lcurand
SRCS = $(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o)
PROFILES=$(OBJS:.o=.profile)

all: $(OBJS)

%.o:%.cpp Makefile
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)


clean:
	rm  -f *.o *.out
