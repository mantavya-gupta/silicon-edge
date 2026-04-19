CXX = g++
CXXFLAGS = -O3 -std=c++20 -march=native

BUILD = build
SRC = src/engine

all: $(BUILD)/silicon_edge

$(BUILD)/silicon_edge: $(SRC)/loader.cpp $(SRC)/main.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD):
	mkdir -p $(BUILD)

bench: all
	python3 tools/gen_weights.py
	python3 benchmarks/run_bench.py

clean:
	rm -rf $(BUILD) weights/*.bin benchmarks/results.png
