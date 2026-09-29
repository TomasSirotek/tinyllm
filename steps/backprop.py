w      = [5.0, -2.0, 0.0]
target = [3.0,  1.0, 4.0]

def loss(w):
    # total squared error across all three
    return sum((w[i] - target[i]) ** 2 for i in range(len(w)))

learning_rate = 0.1
eps = 0.0001

for step in range(50):
    grads = []
    for i in range(len(w)):          # one nudge per parameter
        w[i] += eps; up   = loss(w)
        w[i] -= 2*eps; down = loss(w)
        w[i] += eps                  # put it back where it was!
        grads.append((up - down) / (2 * eps))

    for i in range(len(w)):          # now move every parameter
        w[i] = w[i] - learning_rate * grads[i]

    if step % 10 == 0:
        print(f"step {step:2d}  w={[round(v,3) for v in w]}  loss={loss(w):.6f}")

print("final w:", [round(v, 3) for v in w])
































