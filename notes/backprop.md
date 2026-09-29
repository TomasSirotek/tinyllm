4. Gradients - backprop.py (how a model actually learns)
  
  - Backprop is the tuning.
  - Its not writing the 4225 numbers. 
  - Starting random and let the model fix them itself

  - 1. guess - predict next letter
  - 2. Score - loss = how wrong it was. 1 numbers
  - 3. For each of the 4425 numbers - would nudging this UP make the loss better/worse.
       That answers is gradient. 
  - 4. Nudge every number in the better direction, slightly.
  - 5. Repeat 3k times.

	4.1 What a gradient is
	     "If I nudge this number up a little, does the loss go up or down?"
	     That is all. You can find out by just trying it.

		4.1.1 Loss = how wrong we are. We used (w - target)^2.
		      w=5, target=3  ->  loss = (5-3)^2 = 4

		4.1.2 Nudge w up a hair and down a hair, compare:

			up   = loss(5.0001) = 4.0004
			down = loss(4.9999) = 3.9996
			gradient = (up - down) / (2 * 0.0001) = 4.0

		4.1.3 Gradient is POSITIVE = going up made it worse.
		      So go the other way. Subtract it.

	4.2 The one line that trains everything

			w = w - learning_rate * grad

	     Gradient gives the direction. Learning rate decides how big a step.
	     In PyTorch: loss.backward() finds the direction, optimizer.step() moves.

	4.3 It self-regulates
	     Far from the answer -> big gradient -> big step.
	     Close to the answer -> small gradient -> small step.
	     Nobody coded that. It falls out of the shape of the loss.

	4.4 What a bad learning rate looks like (lr = 1.5)

			w: 5 -> -1 -> 11 -> -13 -> 35 -> -61 -> ...

		4.4.1 The direction was RIGHT every single step. The step was too big,
		      so it flew past 3 and landed further away on the other side.
		      Each round trip doubles the damage.

		4.4.2 That is all "learning rate too high" ever means.

	4.5 The grad = 0.0000 moment (my run)

			step 40  w=-1803881519820  loss=3.25e24  grad=0.0000
			step 45  w=-1803881519820  loss=3.25e24  grad=0.0000

		4.5.1 w stopped moving. Looks converged. It is the worst answer it ever had.

		4.5.2 Why: loss was 3.25e24. A float64 only holds ~16 digits, so numbers
		      that big cannot store a tiny change. up and down came out as the
		      SAME number, so up - down = 0.

		4.5.3 Lesson: "loss stopped changing" does NOT mean "it worked".

	4.6 Why real models do not nudge
	     Nudging is perfect for understanding a gradient, but:
	       - it breaks at extreme values (4.5)
	       - it needs 2 loss calculations per parameter
	     GPT-3 has 175 billion parameters. Calculus (backward()) gets them all
	     exactly, in one pass. Same number, better method.

	4.7 Many parameters at once

		4.7.1 w = [5, -2, 0]  ->  target = [3, 1, 4]. Same loop, nudge each in turn.

		4.7.2 First step, each one moved a different amount:

			5.0  -> 4.6    moved 0.4   (was 2 away)
			-2.0 -> -1.4   moved 0.6   (was 3 away)
			0.0  -> 0.8    moved 0.8   (was 4 away)

		      Distance 2,3,4 -> movement 0.4,0.6,0.8. Proportional.

		4.7.3 The three numbers never talk to each other. Each only knows its own
		      nudge. They still all arrive together.
		      That is how a billion-parameter model trains: every parameter
		      independently asking "which way is downhill for me?"

	4.8 Gotcha worth remembering
	     After nudging w[i], put it back before nudging w[i+1], and collect ALL
	     gradients before moving ANY of them. Otherwise you are measuring a
	     moving target.
