"""
Step 11. Give a friend their own SympAI key, list who has one, or take one back.

    uv run python scripts/friends.py add asha
    uv run python scripts/friends.py list
    uv run python scripts/friends.py remove asha
    uv run python scripts/friends.py remove asha --forget-chart

Add yourself as "me" so the chat screen and the terminal share your chart.

The key is printed once. Only its hash is saved in data/friends.json, so if a
friend loses it, remove them and add them again for a new one. A change here
takes effect on the next request, with no restart.
"""

import argparse
import sys

from sympai.auth import DAILY_MESSAGES, OWNER, add_friend, friends_file, load_friends, remove_friend


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage SympAI keys for friends.")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="make a key for someone")
    add.add_argument("name")
    commands.add_parser("list", help="show who has a key")
    remove = commands.add_parser("remove", help="take a key back")
    remove.add_argument("name")
    remove.add_argument("--forget-chart", action="store_true", help="also delete their saved chart")
    options = parser.parse_args()

    if options.command == "add":
        try:
            key = add_friend(options.name)
        except ValueError as error:
            sys.exit(str(error))
        name = options.name.strip().lower()
        print(f"\nKey for {name}, shown only this once:\n\n    {key}\n")
        print("Send it to them privately, with the link to SympAI's page. They paste the key on the welcome screen.")
        print("    The page         your Vercel link, or http://localhost:3000 on this laptop (README, step 12)")
        if name != OWNER:
            print(f"\nThey can send {DAILY_MESSAGES} messages a day. Their chart is kept on this laptop, under the name {name}.")
    elif options.command == "list":
        friends = load_friends()
        if not friends:
            print(f"No keys yet. Keys live in {friends_file()}.")
        for name, entry in sorted(friends.items()):
            print(f"{name:20} added {entry.get('added', '?')}")
    elif options.command == "remove":
        if not remove_friend(options.name):
            sys.exit(f"No key for {options.name}.")
        print(f"Removed {options.name}. Their key stops working on their next message.")
        if options.forget_chart:
            from sympai import chart as charts

            charts.forget(charts.open_chart_store(), options.name.strip().lower())
            print("Their chart is deleted too.")


if __name__ == "__main__":
    main()
