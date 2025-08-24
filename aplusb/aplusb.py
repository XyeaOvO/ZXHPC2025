import sys

def main():
    input_data = sys.stdin.read().strip()
    A, B = map(int, input_data.split())
    print(A + B)

if __name__ == "__main__":
    main()
