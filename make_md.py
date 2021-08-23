table = [['' for j in range(10)] for i in range(10)]

for i in range(1, 10):
    table[i][0] = f'![taj{i:03d}](taj/taj{i:03d}_resized.png)'

for j in range(1, 10):
    table[0][j] = f'![taj{j:03d}](taj/taj{j:03d}_resized.png)'

for i in range(1, 10):
    for j in range(1, 10):
        table[i][j] = f'![{i:03d}_{j:03d}](taj/taj{i:03d}_taj{j:03d}.png)'

with open('taj.md', 'w', encoding='utf8') as file:
    file.write(' | '.join([''] + table[0]) + '\n')
    file.write(' | '.join([''] + ['---'] * 10) + '\n')
    for row in table[1:]:
        file.write(' | '.join([''] + row) + '\n')
