
# python3 blowfish_demo.py --key TestKey01 --pt 0123456789ABCDEF

import argparse
import sys
from textwrap import dedent

# Part A: Blowfish (library)
def real_blowfish_ecb(key_bytes: bytes, pt_hex: str):
    try:
        from Crypto.Cipher import Blowfish
    except ImportError:
        print("Missing dependency: pycryptodome")
        print("Install with: pip install pycryptodome")
        sys.exit(1)

    pt = bytes.fromhex(pt_hex)
    if len(pt) != 8:
        raise ValueError("This demo expects exactly 8 bytes (64 bits) of plaintext for one-block ECB.")

    cipher = Blowfish.new(key_bytes, Blowfish.MODE_ECB)
    ct = cipher.encrypt(pt)

    decipher = Blowfish.new(key_bytes, Blowfish.MODE_ECB)
    dec = decipher.decrypt(ct)

    return pt, ct, dec


# Part B: EDUCATIONAL round-by-round tracer
# (Not full Blowfish: simplified S-boxes, no key expansion)
P_INIT = [
    0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
    0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
    0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
    0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917,
    0x9216D5D9, 0x8979FB1B
]

# Simplified S-boxes just to show mechanics (NOT official Blowfish S-boxes)
S = [
    [(i * 0x1010101) & 0xFFFFFFFF for i in range(256)],
    [((i + 1) * 0x01010101) & 0xFFFFFFFF for i in range(256)],
    [((i + 2) * 0x00100101) & 0xFFFFFFFF for i in range(256)],
    [((i + 3) * 0x00010101) & 0xFFFFFFFF for i in range(256)],
]

def F_demo(x: int) -> int:
    a = (x >> 24) & 0xFF
    b = (x >> 16) & 0xFF
    c = (x >> 8) & 0xFF
    d = x & 0xFF
    # modular arithmetic with 32-bit wraparound
    return (((S[0][a] + S[1][b]) & 0xFFFFFFFF) ^ S[2][c] + S[3][d]) & 0xFFFFFFFF

def encrypt_block_trace(pt_hex: str, verbose_rounds: bool = True):
    block = int(pt_hex, 16)
    L = (block >> 32) & 0xFFFFFFFF
    R = block & 0xFFFFFFFF

    lines = []
    lines.append("EDUCATIONAL TRACE (shows Feistel structure; NOT the production ciphertext)")
    lines.append(f"Initial split:")
    lines.append(f"  L0 = 0x{L:08X}")
    lines.append(f"  R0 = 0x{R:08X}\n")

    for i in range(16):
        # round function
        L ^= P_INIT[i]
        R ^= F_demo(L)
        # swap
        L, R = R, L

        if verbose_rounds:
            lines.append(f"After round {i+1:02d}:")
            lines.append(f"  L{i+1} = 0x{L:08X}")
            lines.append(f"  R{i+1} = 0x{R:08X}\n")

    # undo final swap
    L, R = R, L
    # final whitening
    R ^= P_INIT[16]
    L ^= P_INIT[17]

    lines.append("After final whitening:")
    lines.append(f"  L16 = 0x{L:08X}")
    lines.append(f"  R16 = 0x{R:08X}\n")

    ct_demo = ((L << 32) | R) & 0xFFFFFFFFFFFFFFFF
    lines.append(f"Ciphertext (educational) = 0x{ct_demo:016X}")
    return "\n".join(lines), ct_demo


# Helpers
def header(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

def main():
    parser = argparse.ArgumentParser(
        description="Blowfish demo: real encryption + educational internal trace"
    )
    parser.add_argument("--key", type=str, default="TestKey01", help="ASCII key string (default: TestKey01)")
    parser.add_argument("--pt", type=str, default="0123456789ABCDEF", help="64-bit plaintext hex (16 hex chars)")
    parser.add_argument("--no-trace", action="store_true", help="Disable round-by-round educational trace output")
    args = parser.parse_args()

    key_bytes = args.key.encode("utf-8")
    pt_hex = args.pt.strip().upper()

    if len(pt_hex) != 16 or any(c not in "0123456789ABCDEF" for c in pt_hex):
        print("Error: --pt must be exactly 16 hex characters (64 bits). Example: 0123456789ABCDEF")
        sys.exit(1)

    # Part A: Real Blowfish
    header("PART A — REAL BLOWFISH (PyCryptodome, ECB, 1 block)")
    pt, ct, dec = real_blowfish_ecb(key_bytes, pt_hex)

    print(f"Key (ASCII):   {args.key}")
    print(f"Plaintext:     {pt.hex().upper()}")
    print(f"Ciphertext:    {ct.hex().upper()}")
    print(f"Decrypted:     {dec.hex().upper()}")

    if dec != pt:
        print("\nERROR: decryption did not match plaintext (should never happen).")
        sys.exit(1)

    print("\n✔ Verified: D_K(C) = M (decryption recovers original plaintext)")

    # Part B: Educational trace
    if not args.no_trace:
        header("PART B — EDUCATIONAL INTERNAL TRACE (Feistel rounds)")
        trace_text, ct_demo = encrypt_block_trace(pt_hex, verbose_rounds=True)
        print(trace_text)

if __name__ == "__main__":
    main()