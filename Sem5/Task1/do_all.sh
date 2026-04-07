clang main.c -o main

echo "calculating data..."
./main

echo "generating graph..."
python3 graph_gen.py
