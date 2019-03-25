with open('prediction_test.log', 'r') as f:
    lines = f.readlines()
lines = [x.strip() for x in lines]

wins = 0
loses = 0
for line in lines:
    win = int(line.split()[1][:-1])
    lose = int(line.split()[3][:-1])
    wins += win
    loses += lose
print('Wins: %d, loses: %d. Win rate: %4f' % (wins, loses, wins/(wins+loses)))
