clang main.c -o main

echo "calculating data..."
./main > data/result.csv

echo "generating graph..."
python3 graph_gen.py
