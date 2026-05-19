from clocktower_img2json.data import OfficialRole
from clocktower_img2json.parser import OCRLine, parse_script_lines


def test_parse_script_lines_official_roles():
    official_by_name = {
        "washerwoman": OfficialRole(
            id="washerwoman",
            name="Washerwoman",
            team="townsfolk",
            ability="You start knowing that 1 of 2 players is a particular Townsfolk.",
        )
    }

    lines = [
        OCRLine("Typhon with a Gap by Manticor", 10, 10, 300, 40),
        OCRLine("TOWNSFOLK", 10, 60, 140, 80),
        OCRLine("Washerwoman", 20, 100, 200, 120),
        OCRLine("You start knowing that 1 of 2 players is a particular Townsfolk.", 20, 130, 420, 150),
    ]

    script_name, author, roles = parse_script_lines(lines, official_by_name)

    assert script_name == "Typhon with a Gap"
    assert author == "Manticor"
    assert len(roles) == 1
    assert roles[0].official is not None
    assert roles[0].official.id == "washerwoman"
