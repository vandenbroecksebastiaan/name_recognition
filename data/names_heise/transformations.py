with open("nam_dict.txt", "rb") as text:
    temp = text.read()
    temp = temp.split(b"\n")

# Strip the data
temp = [i[2:-3] for i in temp]

# Find out where the data starts
for idx, line in enumerate(temp):
    if b"begin of name list" in line:
        data_idx = idx
    if b"list of countries" in line:
        countries_idx = idx

# Read the index of each line
country_idx_map = {}
i = 1
for idx in range(countries_idx+7, data_idx-19):
    line = temp[idx]
    # These are the lines that contain the countries
    if i % 3 == 1:
        country = line.strip()
    # These are the lines that contain the index
    if i % 3 == 2:
        country_idx = line.find(b"|")
        country_idx_map[country] = country_idx
    i += 1

# Read each line and find out its occurence per country
for idx in range(data_idx+2, data_idx+100):
    print(temp[idx])
