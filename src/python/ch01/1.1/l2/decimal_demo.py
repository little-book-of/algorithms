from decimal import Decimal, getcontext

def main():
    # Set more precision than default to illustrate exact decimal sums
    getcontext().prec = 28

    a = Decimal('0.1') + Decimal('0.2')
    print("Decimal('0.1') + Decimal('0.2') =", a)
    print("Equal to Decimal('0.3')?", a == Decimal('0.3'))

    # Contrast with binary float printed via Decimal for readability
    from_float = Decimal(0.1) + Decimal(0.2)
    print("Decimal(0.1) + Decimal(0.2) =", from_float)  # shows binary float -> decimal conversion artifact

if __name__ == "__main__":
    main()
