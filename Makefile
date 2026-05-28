# cai - C AI trainer (220k GB300 scaffold). GPU-free build by default.
# Set CUDA=1 to compile the (stubbed) CUDA backend path with -DCAI_WITH_CUDA.

CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -g -Wall -Wextra
CFLAGS  += -Iinclude -Isrc -MMD -MP
LDLIBS  += -lm

ifeq ($(CUDA),1)
CFLAGS  += -DCAI_WITH_CUDA
endif

BUILD   := build
BIN     := bin
LIB_SRC := $(wildcard src/*.c)
LIB_OBJ := $(patsubst src/%.c,$(BUILD)/%.o,$(LIB_SRC))
LIB     := $(BUILD)/libcai.a

TOOLS   := plan_compiler topology_linter trainer reftrain plan_verify plan_recover
TOOLBIN := $(addprefix $(BIN)/,$(TOOLS))
TESTBIN := $(BIN)/cai_test

.PHONY: all lib tools test demo clean
all: lib tools $(TESTBIN)
lib: $(LIB)
tools: $(TOOLBIN)

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(LIB_OBJ)
	ar rcs $@ $^

$(BIN)/%: tools/%.c $(LIB) | $(BIN)
	$(CC) $(CFLAGS) $< $(LIB) -o $@ $(LDLIBS)

$(TESTBIN): tests/cai_test.c $(LIB) | $(BIN)
	$(CC) $(CFLAGS) $< $(LIB) -o $@ $(LDLIBS)

test: $(TESTBIN)
	./$(TESTBIN)

# End-to-end: everything that runs with no GPU.
demo: tools
	@echo "== lint full220k =="
	./$(BIN)/topology_linter configs/full220k.cfg
	@echo "== verify all 220,032 ranks pre-launch =="
	./$(BIN)/plan_verify configs/full220k.cfg
	@echo "== compile 220,032-GPU sample =="
	./$(BIN)/plan_compiler configs/full220k.cfg out/full
	@echo "== replay stage-0 of the 220k plan =="
	./$(BIN)/trainer out/full/plan.rank000000.bin out/full/topology.bin --steps 5
	@echo "== actually train a real transformer on CPU =="
	./$(BIN)/reftrain --steps 400

$(BUILD) $(BIN):
	mkdir -p $@

clean:
	rm -rf $(BUILD) $(BIN) out

-include $(LIB_OBJ:.o=.d)
