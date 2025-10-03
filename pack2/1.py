def parser(line: str) -> set[str]:
    punct = ".,!?;:-()[]{}\"'«»"
    clean = "".join(ch for ch in line if ch not in punct)
    words = clean.lower().split()
    return set(words)

line = "Привет, купи наш товар! Купи наш товар..."
print(parser(line))   # {'наш', 'товар', 'привет', 'купи'}
