with open("data/names_heise/nam_dict.txt", "rb") as text:
    temp = text.read()
    temp = temp.split(b"\n")

# Strip the data
temp = [i[3:-3] for i in temp]

# Find out where the data starts
for idx, line in enumerate(temp):
    if b"begin of name list" in line:
        data_idx = idx
        data_idx += 2
    if b"list of countries" in line:
        countries_idx = idx
        countries_idx += 7

# Read the index of each line
idx_country_map = {}
for counter, idx in enumerate(range(countries_idx, data_idx-19)):
    line = temp[idx]
    # These are the lines that contain the countries
    if counter % 3 == 0:
        country = line.strip()
        country = country.decode("iso8859-1")
        country = "".join(i for i in country if i.isalnum())
    # These are the lines that contain the index
    if counter % 3 == 1:
        country_idx = line.find(b"|")
        idx_country_map[country_idx] = country

# For every line we are going to need to read the name
# Next, we will search every number and its index
byte_str_numbers = [b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9"]
name_importance_map = {}
for idx in range(data_idx, len(temp)-1):
    line = temp[idx]
    name = line[:10]
    name = name.decode("iso8859-1")
    name = "".join(i for i in name if i.isalnum())
    name = name.strip()

    country_importance = []

    # For every number that is encoded as a byte string, we are going to
    # check if it is in the line. If it is in the line, we are going to
    # take note of what country the position of the number represents.
    # The number itself that is in the line represents the importance of
    # the name for the country.
    for number in byte_str_numbers:
        country_idx = line.find(number)
        importance = int(number)
        if country_idx != -1:
            country = idx_country_map[country_idx]
            country_importance.append([country, importance])

    name_importance_map[name] = country_importance

# Lastly, write the names to a file
with open("transformed_names_heise.txt", "w") as file:
    for i, j in name_importance_map.items():
        occurences = "".join([str(k) for k in j])
        file.write(i + "|[" + occurences + "]\n")
