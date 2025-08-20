# OCRC, a AI for optical character recognition written in C
# Copyright (C) 2023-2025 João Edison Roso Manica
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

OPTIMIZATION =
# In RaspRaspberry Pi O/S or in Debian distros replace -l:libblas.a with -latlas
CFLAGS = -Wall -Wno-implicit -lm -lpng -l:libblas.a -g

BUILD_DIR := ./build

vision_lib := $(BUILD_DIR)/lib/vision.o
neural_net_lib := $(BUILD_DIR)/lib/neural_net.o
model_obj := $(BUILD_DIR)/lib/model_obj.o

PROGRAMS := view init_net training
BINS := $(PROGRAMS:%=$(BUILD_DIR)/%)

all: $(vision_lib) $(neural_net_lib) $(model_obj) $(BINS)

$(vision_lib): src/vision.c
	mkdir -p $(BUILD_DIR)/lib
	gcc -c $< -o $@ -Wall -Wno-implicit $(OPTIMIZATION) -g
$(neural_net_lib): src/neural_net.c
	mkdir -p $(BUILD_DIR)/lib
	gcc -c $< -o $@ -Wall -Wno-implicit $(OPTIMIZATION) -g
$(model_obj): src/model.c
	mkdir -p $(BUILD_DIR)/lib
	gcc -c $< -o $@ -Wall -Wno-implicit $(OPTIMIZATION) -g

$(BUILD_DIR)/init_net: src/init_net.c $(neural_net_lib)
	mkdir -p $(BUILD_DIR)
	$(CC) $^ -o $@ $(CFLAGS) $(OPTIMIZATION)

$(BUILD_DIR)/view: src/view.c $(neural_net_lib) $(vision_lib) $(model_obj)
	mkdir -p $(BUILD_DIR)
	$(CC) $^ -o $@ $(CFLAGS) $(OPTIMIZATION)
$(BUILD_DIR)/training: src/training.c $(neural_net_lib) $(vision_lib) $(model_obj)
	mkdir -p $(BUILD_DIR)
	$(CC) $^ -o $@ $(CFLAGS) $(OPTIMIZATION) -lpthread

clean:
	rm -rd $(BUILD_DIR)
