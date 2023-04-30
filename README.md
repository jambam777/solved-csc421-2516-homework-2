Download Link: https://assignmentchef.com/product/solved-csc421-2516-homework-2
<br>
<strong>Submission: </strong>You must submit two files through MarkUs<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>:

<ul>

 <li>A PDF of your solutions. You can produce the file however you like (e.g. LaTeX, Microsoft Word, scanner), as long as it is readable.</li>

 <li>Your completed py</li>

</ul>

<strong>Late Submission: </strong>MarkUs will remain open until 3 days after the deadline, after which no late submissions will be accepted. The late penalty is 10% per day, rounded up.

Weekly homeworks are individual work. See the Course Information handout<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> for detailed policies.

<ol>

 <li><strong>MAML. [5pts] </strong>This question is meant to introduce a cool application of automatic differentiation. Much like gradient-based hyperparameter optimization (last two slides of Lecture 6), it involves treating the gradient descent learning procedure itself as a computation graph, and differentiating through it. While the algorithm would be a major pain to implement by hand, it only requires a few extra lines of Autograd code compared to ordinary neural net training.</li>

</ol>

Suppose you want to train an agent that learns to perform many different but related tasks, such as having a robot arm pick up a variety of objects. The agent, through gaining experience with many such tasks, ought to be able to improve the rate at which it can learn similar tasks. This kind of learning is known as <strong>learning to learn</strong>, or <strong>meta-learning</strong>.

This question concerns a meta-learning algorithm called <strong>Model-Agnostic Meta-Learning (MAML, pronounced “mammal”)</strong>.<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> The idea is that if you choose a good enough set of initial weights for the network, it should be possible to learn a new task in only a few steps of gradient descent. Hence, MAML trains a single, task-generic set of weights, with the meta-objective defined as the loss on any particular task after <em>K </em>steps of gradient descent. (<em>K </em>is a small number, such as 5.) The term “model-agnostic” is because MAML assumes pretty much nothing about the model, other than that it’s trainable by gradient escent.

MAML was originally formulated in the more complex setting of reinforcement learning, but we will consider the setting of simple univariate regression problems. We will sample random univariate regression problems, where the inputs are sampled uniformly from the interval [−3<em>,</em>3], and the functions are sampled as random piecewise constant functions with breaks at the integers. For instance,

It’s worth thinking a bit about how the meta-learner might solve this problem. Clearly, 5 iterations would not be enough to train a generic MLP from scratch. But suppose you initialized the network such that each hidden unit computed a basis function which takes the value 1 on the interval [<em>k,k </em>+ 1] for some integer <em>k</em>, and 0 everywhere else. Then you could fit the function simply by adjusting the weights in the output layer, which is just a linear regression problem, and can therefore be solved pretty quickly. Hence, such a network would be a great according to MAML’s objective. Of course, there are probably other good ways for MAML to solve this problem, and we don’t know what it will actually do when we run it.

The code you actually have to write is mostly straightforward once you know how to use Autograd. The challenging (and hopefully valuable) part is wrapping your head around how the starter code works. This is defined in maml.py. Here are the functions and classes it defines:

<ul>

 <li>net_predict: this implements the forward pass for an MLP. The parameters are stored in a Python dict params.</li>

 <li>random_init: initializes the weights to a Gaussian with a small standard deviation.</li>

 <li>ToyDataGen: This class generates the random piecewise linear functions.</li>

 <li>gd_step: Performs one step of gradient descent. Note that this function returns a new set of parameters, rather than modifying the arrays that were passed in.</li>

 <li>InnerObjective: The cost function for <em>one </em>regression dataset. This is just mean squared error.</li>

 <li>MetaObjective: The MAML objective, i.e. the inner cost after num_steps steps of gradient descent.</li>

 <li>train: Runs the actual training, i.e. repeatedly samples random regression datasets and does gradient descent on the meta-objective.</li>

</ul>

Here is what you need to do. Each of these parts requires only a few lines of code, and you should not need to do any messy derivations.

<ul>

 <li><strong>[2pts] </strong>Implement gd_step. You should do this by calling grad.</li>

 <li><strong>[2pts] </strong>Implement __call__. (This is Python syntax for the method that gets called when you call the class instance as if it were a function, i.e. meta_obj(params).) Your implementation should call gd_step.</li>

 <li><strong>[1pt] </strong>Finish the implementation of train. I.e., sample a random regression dataset, and do a gradient descent step on the meta-objective.</li>

</ul>

Once you finish the code, calling train will produce a visualization such as the following, where the thinnest line corresponds to the initial parameters learned by MAML, and thicker lines correspond to more steps of SGD:

Observe that your solution will involve calling gd_step on a function which itself calls gd_step. Since gd_step calls ag.grad, this means you are calling ag.grad on a computation graph which was itself generated by ag.grad. Understanding why this happens is an important part of understanding the code.

Submit your code solution as maml.py.    You don’t need to submit anything else for this question.

<ol start="2">

 <li><strong>Adam. [5pts] </strong>Adam<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a> is a widely used optimization algorithm which essentially combines the benefits of RMSprop and momentum. Here is a slightly simplified version of the original algorithm. All arithmetic operations, such as squaring or division, are applied elementwise.</li>

</ol>

<strong>m</strong><sub>0 </sub>← 0 <strong>v</strong><sub>0 </sub>← 0 <em>t </em>← 0

While <em>θ<sub>t </sub></em>not converged: <strong>g</strong><em>t </em>← ∇J(<em>θ</em><em>t</em>−1) <strong>m</strong><em><sub>t </sub></em>← <em>β</em><sub>1</sub><strong>m</strong><em><sub>t</sub></em>−<sub>1 </sub>+ (1 − <em>β</em><sub>1</sub>)<strong>g</strong><em><sub>t </sub></em><strong>v</strong><em>t </em>← <em>β</em>2<strong>v</strong><em>t</em>−1 + (1 − <em>β</em>2)<strong>g</strong><em>t</em>2

The hyperparameters of the algorithm are the learning rate <em>α</em><sub>A</sub>, the moments timescales <em>β</em><sub>1 </sub>and <em>β</em><sub>2</sub>, and the damping term <sub>A</sub>. The <em>A </em>subscript stands for “Adam,” to distinguish these hyperparameters from the other algorithms discussed in this question.

Here is what you need to analyze:

<ul>

 <li><strong>[1pt] </strong>Recall the RMSprop algorithm, rewritten here to match the notation of this question:</li>

</ul>

<strong>v</strong><sub>0 </sub>← 0 <em>t </em>← 0

While <em>θ<sub>t </sub></em>not converged: <strong>g</strong><em>t </em>← ∇J(<em>θ</em><em>t</em>−1) <strong>v</strong><em>t </em>← <em>γ</em><strong>v</strong><em>t</em>−1 + (1 − <em>γ</em>)<strong>g</strong><em>t</em>2

The hyperparameters are <em>α</em><sub>R</sub>, <em>γ</em>, and <sub>R</sub>. Specify Adam hyperparameters ( which make Adam equivalent to RMSprop with hyperparameters (). You should explain your answer, though a full derivation isn’t required.

<ul>

 <li><strong>[2pts] </strong>Now consider SGD with momentum: <strong>p</strong><sub>0 </sub>← 0 <em>t </em>← 0</li>

</ul>

While <em>θ<sub>t </sub></em>not converged:

<strong>p</strong><em><sub>t </sub></em>← <em>µ</em><strong>p</strong><em><sub>t</sub></em><sub>−1 </sub>− (1 − <em>µ</em>)∇J(<em>θ<sub>t</sub></em><sub>−1</sub>) <em>θ</em><em>t </em>← <em>θ</em><em>t</em>−1 + <em>α</em>S<strong>p</strong><em>t</em>

Specify Adam hyperparameters () which make Adam approximately equivalent to momentum SGD with parameters (<em>α</em><sub>S</sub><em>,µ</em>). Explain your answer.

<ul>

 <li><strong>[2pts] </strong>An important fact about Adam is that it is invariant to rescaling of the loss function. I.e., suppose we have a loss function L(<em>y,t</em>), and we define a modified loss function as L˜(<em>y,t</em>) = <em>C </em> L(<em>y,t</em>) for some positive constant <em>C</em>. Show that for = 0, Adam is invariant to this rescaling, i.e. it passes through the same sequence of iterates <em>θ</em><sub>0</sub><em>,…,</em><em>θ<sub>T</sub></em>.</li>

</ul>

<em>Hint: Denote the quantities computed by Adam on the modified loss function as </em><strong>g</strong>˜<em><sub>t</sub>, </em><strong>m</strong>˜ <em><sub>t</sub>, etc. Use induction to find relationships between these and the original </em><strong>g</strong><em><sub>t</sub>, </em><strong>m</strong><em><sub>t</sub>, etc.)</em>

<a href="#_ftnref1" name="_ftn1">[1]</a> <a href="https://markus.teach.cs.toronto.edu/csc421-2019-01">https://markus.teach.cs.toronto.edu/csc421-2019-01</a>

<a href="#_ftnref2" name="_ftn2">[2]</a> <a href="http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/syllabus.pdf">http://www.cs.toronto.edu/</a><a href="http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/syllabus.pdf">~</a><a href="http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/syllabus.pdf">rgrosse/courses/csc421_2019/syllabus.pdf</a>

<a href="#_ftnref3" name="_ftn3">[3]</a> You’re welcome to read the original paper, though this isn’t necessary to do this question: <a href="https://arxiv.org/abs/1703.03400">https://arxiv.org/ </a><a href="https://arxiv.org/abs/1703.03400">abs/1703.03400</a>

<a href="#_ftnref4" name="_ftn4">[4]</a> Here is the original paper, but you don’t need to read it to solve this question: <a href="https://arxiv.org/abs/1412.6980">https://arxiv.org/abs/1412. </a><a href="https://arxiv.org/abs/1412.6980">6980</a>