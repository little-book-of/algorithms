from dataclasses import dataclass

# Money as fixed-point cents (int64 policy). Safe up to about ±9e16 cents.
def parse_dollars_to_cents(s: str) -> int:
    """
    Parse strings like '123', '123.4', '123.45', '-0.07' to cents (int).
    Truncates extra fractional digits beyond 2 (policy: banker-friendly).
    """
    s = s.strip()
    sign = -1 if s.startswith('-') else 1
    if s[0] in '+-':
        s = s[1:]
    if '.' in s:
        whole, frac = s.split('.', 1)
    else:
        whole, frac = s, ''
    if not whole:
        whole = '0'
    if not whole.isdigit() or (frac and not frac.isdigit()):
        raise ValueError(f"bad money literal: {s}")
    frac = (frac + '00')[:2]  # pad/truncate to 2 decimals
    return sign * (int(whole) * 100 + int(frac))

def format_cents(c: int) -> str:
    sign = '-' if c < 0 else ''
    c = abs(c)
    return f"{sign}{c//100}.{c%100:02d}"

@dataclass
class Ledger:
    """In-memory ledger with int cents; raises on overdraft by default."""
    balance_cents: int = 0

    def deposit(self, cents: int) -> None:
        # No overflow in Python, but you can bound-check here if porting.
        self.balance_cents += cents

    def withdraw(self, cents: int) -> None:
        if cents > self.balance_cents:
            raise ValueError("insufficient funds")
        self.balance_cents -= cents

    def transfer_to(self, other: "Ledger", cents: int) -> None:
        self.withdraw(cents)
        other.deposit(cents)
