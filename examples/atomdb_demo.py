"""Browse an existing local AtomDB; no data downloads or setup are run.

After installation:
  python examples/atomdb_demo.py /path/to/atomdb --temperature 0.2
"""
import argparse

from sherpa_plasma import AtomDB, LineID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", help="AtomDB root (defaults to $ATOMDB)")
    parser.add_argument("--temperature", type=float, default=.2, help="plasma kT in keV")
    args = parser.parse_args()
    with AtomDB(args.root) as db:
        print("Database:", db.info)
        print("Atomic O VII transitions, 21.5-22.2 Angstrom:")
        print(db.find_lines(21.5, 22.2, element="O", ion=7))
        rows = db.find_lines(21.5, 22.2, element="O", ion=7,
                             kT=args.temperature, brightest=3)
        print("Strongest at", args.temperature, "keV:")
        print(rows)
        if len(rows):
            selected = LineID.from_row(rows[0])
            print("Strongest-line emissivity:", db.line_emissivity(selected, args.temperature),
                  "photons cm^3/s")
        if len(rows) > 1:
            print("Ratio of two strongest:", db.ratio_emissivity(
                LineID.from_row(rows[0]), LineID.from_row(rows[1]), args.temperature))


if __name__ == "__main__":
    main()
