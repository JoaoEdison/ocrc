# OCRC, a AI for optical character recognition writed in C
# Copyright (C) 2023-2023 João Edison Roso Manica
#
# OCRC is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OCRC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

OPTIMIZATION = -Og
CFLAGS = -Wall -Wno-implicit -lm -lpng -lcblas -pg

LIB = neural_img.c
LIB_OBJECT = $(LIB:.c=.o)

PROGRAM1 = view.out
PROGRAM1_SOURCE = view.c

PROGRAM2 = init_net.out
PROGRAM2_SOURCE = init_net.c

PROGRAM3 = training.out
PROGRAM3_SOURCE = training.c

all: $(PROGRAM1) $(PROGRAM2) $(PROGRAM3)

$(LIB_OBJECT):
	gcc -c $(LIB) -Wall -Wno-implicit -Wno-multistatement-macros $(OPTIMIZATION)

$(PROGRAM1): $(LIB_OBJECT)
	gcc $(PROGRAM1_SOURCE) $(LIB_OBJECT) -o $(PROGRAM1) $(CFLAGS) $(OPTIMIZATION)

$(PROGRAM2): $(LIB_OBJECT)
	gcc $(PROGRAM2_SOURCE) $(LIB_OBJECT) -o $(PROGRAM2) $(CFLAGS) $(OPTIMIZATION)

$(PROGRAM3): $(LIB_OBJECT)
	gcc $(PROGRAM3_SOURCE) $(LIB_OBJECT) -o $(PROGRAM3) $(CFLAGS) -lpthread $(OPTIMIZATION)
