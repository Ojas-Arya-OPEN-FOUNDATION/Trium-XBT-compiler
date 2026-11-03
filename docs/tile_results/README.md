# Tile survey results

Drop `tile_survey` CSV output here, one file per machine:

```
g++ -O3 -march=native -std=c++20 -I. -Iinclude tools/tile_survey.cpp -o tile_survey
./tile_survey --sizes=256,512,1024 --reps=11 > docs/tile_results/<cpu>-<compiler>.csv
```

Name the file after the microarchitecture and compiler, e.g.
`xeon-gold-6240-gcc12.csv`, `ryzen-7950x-clang17.csv`, `apple-m2-clang15.csv`.

The tile default is only changed when a tile wins on **every** machine for the
ISA that `Isa::Auto` selects there. See `../TILE_DEFAULT.md`.

Rules for a usable submission:

- At least 11 in-process reps; report the median column.
- Run the whole invocation at least 5 times and paste every run - run-to-run
  spread on shared/virtualised cores is the same size as the effect measured.
- Verify before trusting: any row whose status is not `ok` is meaningless.
- Record the CPU, core count, whether it is virtualised/shared, and the exact
  compiler version and flags.
