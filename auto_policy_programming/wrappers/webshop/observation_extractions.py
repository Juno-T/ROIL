import re

re_sep = '\s*\[SEP\]\s*'

def extract_observation(text, state_name, button_list: list = []):
    observations = {}
    # Use regular expressions to find instructions
    instructions_match = re.search(rf'Instruction:{re_sep}(.+?){re_sep}', text)
    if instructions_match:
        observations["instruction"] = instructions_match.group(1)
    if state_name == "results":
        # Initialize variables to store extracted data
        items = []

        # Use regular expressions to find items and their details
        item_matches = re.finditer(rf'{re_sep} ([A-Z0-9]+){re_sep}(.+?){re_sep}(\$[\d.]+)', text)
        for match in item_matches:
            item = (match.group(1), match.group(2), match.group(3))
            items.append(item)

        # Store items in the observations dictionary
        observations["items"] = items

    if state_name == "item":
        # item_name = re.search(rf'{re_sep}([A-Za-z\s\'-]+){re_sep}Price:', text)
        item_name = re.search(rf'{re_sep}([^\[]+){re_sep}Price', text, re.IGNORECASE)
        if item_name:
            observations['item_name'] = item_name.group(1).strip()

        price = re.search(rf'Price: (.*?){re_sep}', text, re.IGNORECASE)
        if price:
            observations['price'] = price.group(1).strip()

    return observations
