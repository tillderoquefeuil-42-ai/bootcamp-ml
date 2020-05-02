# bootcamp-ml by 42ai

## Machine Learning MOOC - Stanford


### Matrix

#### Matrices and Vectors
- Matrix :
    - definition : multiple dimensions array (eg: A = [[1, 2, 3], [4, 5, 6]])
    - dimension : number of columns * number of rows (eg: ℝA = 2 * 3 = 6)
    - elements : i, j refers to the i_th columns and j_th row (eg: A₂₁ = 4)
- Vector :
    - definition : matrix that has only 1 column (eg: V = [1, 2, 3, 4])
    - dimension : number of rows (eg: ℝV = 4)
    - elements : i refers to the i_th row (eg: V₂ = 2)

#### Addition and Scalar Multiplication
- Matrix addition/substraction :
    - ONLY WITH Matrices OF THE SAME DIMENSIONS
    - A = [[1, 2, 3], [4, 5, 6]]
    - B = [[9, 8, 7], [6, 5, 4]]
    - A + B = [[10, 10, 10], [10, 10, 10]]
- Matrix scalar multiplication/division :
    - A = [[1, 2, 3], [4, 5, 6]]
    - b = 3
    - A * B = [[3, 6, 9], [12, 15, 18]]

#### Matrix Vector Multiplication
- Matrix vector multiplication :
    - ONLY WITH Matrices OF THE DIMENSIONS MATCHES
    - A = [[1, 2, 3], [4, 5, 6]]
    - B = [2, 3]
    - C = A * B = [14, 19, 24]
    - dimension : ℝA = 2 * 3 & ℝB = 1 * 2 -> ℝC = ℝA₁ * ℝB₀ = 3 * 1
- Simplification : 
    - Prediction = Datamatrix * parameters
    - Datamatrix = [[1, 1, 1, 1], [m₀, m₁, m₂, m₃]]
    - paramters = [θ₀, θ₁]

#### Matrix Matrix Multiplication
- Matrix matrix multiplication :
    - ONLY WITH MATRICES OF THE DIMENSIONS MATCHES
    - A = [[1, 2], [4, 5], [7, 8]]
    - B = [[1, 2, 3], [2, 3, 4]]
    - C = A * B = [[5, 8, 11], [14, 23, 32], [23, 38, 53]]
    - dimension : ℝA = 3 * 2 & ℝB = 2 * 3 -> ℝC = ℝA₁ * ℝB₀ = 3 * 3
- Simplification : 
    - Prediction = Datamatrix * parameters
    - Datamatrix = [[1, 1, 1, 1], [m₀, m₁, m₂, m₃]]
    - paramters = [[h₀θ₀, h₀θ₁], [h₁θ₀, h₁θ₁], [h₂θ₀, h₂θ₁]]

#### Matrix Multiplication Properties
- Not commutative :
    - A * B ≠ B * A
    - C = A * B & D = B * A
    - ℝA = 2 * 3 & ℝB = 3 * 2 
    - -> ℝC = ℝA₀ * ℝB₁ = 2 * 2
    - -> ℝD = ℝB₀ * ℝA₁ = 3 * 3
- Associative
    - (A * B) * C = A * (B * C)
- Identity matrix :
    - I = 1 | [[1, 0], [0, 1]] | [[1, 0, 0], [0, 1, 0], [0, 0, 1]] | etc.
    - For any matrix A, A.I = I.A (with I dimension matches A)

#### Inverse and Transpose
- Matrix inverse :
    - A*A⁻¹ = A⁻¹*A = I
    - 3 * 3⁻¹ = 3⁻¹ * 3 = 3 * 1/3 = 1
- Matrix transpose :
    - inverse rows & columns for a matrix
    - For A with ℝA = m * n & B = At
    - ℝB = n * m & A(ij) = B(ji)


### Machine Learning

#### Supervised learning
- Supervised learning : ("right answers" given) With a given data set and already know what correct output should look like, having the idea that there is a relationship between the input and the output
- Regression problem : predict continuous valued output (straight line/curve line)(eg: price evolution)
- Classification problem : discrete valued output (eg: true/false answers)

#### Unsupervised learning
- Unsupervised learning : data all same label/no labels
- Clustering : separate data in clusters
- Non-clustering : allows you to find structure in a chaotic environment

#### Model Representation
- Data/Training set :
    - m : number of trainnig examples
    - s : input (variable/features)
    - y : output (variable/target)
- Hypothesis : (h) h maps from x's to y's
- Linear regression with one variable/Univariate linear regression : model where h_θ(x) = θ₀ + θ₁ * x

#### Cost Function
- θi : parameters of the model
- define θi :
    - concept : minimize the difference between h(x) and y (in order to get a line as close as possible to dataset)
    - formula : (1/2m) * ∑m,i(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    - Cost function/Squared error function : J(θ₀, θ₁) = (1/2m) * ∑m,i(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    - minimize J(θ₀θ₁)

#### Gradient Descent
- find minimum J(θ₀, θ₁)
- step by step changing θ₀ & θ₁ until finding min value
- formula :
    - θ_j := θ_j - α(∂/∂*θ_j) * J(θ₀, θ₁)
    - α : learning rate (step size)
    - (∂/∂*θ_j) * J(θ₀, θ₁) : derivative (smaller when approach a local minimum)
    - θ_j : with _j = 0 and _j = 1
    - θ_j for θ₀ & θ₁ has to be updated simultaneously

#### Gradient Descent For Linear Regression
- Gradient descent algorithm : θ_j := θ_j - α(∂/∂*θ_j) * J(θ₀, θ₁)
- Linear regression model : 
    - h_θ(x) = θ₀ + θ₁ * x
    - J(θ₀, θ₁) = (1/2m) * ∑m,i(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
- Implementation :
    - global: θ_j := θ_j - α(∂/∂*θ_j) * (1/2m) * ∑m,i(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    - for θ₀: θ₀ := θ₀ - α * (1/m) * ∑m,i(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)
    - for θ₁: θ₁ := θ₁ - α * (1/m) * ∑m,i((h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x⁽ⁱ⁾)

#### Multiple Features
- Notations :
    - n features: x₁, x₂, ... x_n
    - y : result
    - x⁽ⁱ⁾ : vector of all features of the i-th row (x⁽ⁱ⁾_j: value j in x⁽ⁱ⁾)
- Hypothesis : 
    - h_θ(x) = θ₀ + (θ₁ * x₁) + (θ₂ *  x₂) ... + (θ_n *  x_n)
    - x = [x₀, x₁, x₂, ... x_n] (with x₀ = 1)
    - θ = [θ₀, θ₁, θ₂, ... θ_n]
    - θ^T . x (^T: transpose)
- Cost function :
    - J(θ) = (1/2m) * ∑m,i(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
- Gradient Descent :
    - θ_j := θ_j - α(∂/∂*θ_j) * J(θ)
    - for θ_n: θ_n := θ_n - α * (1/m) * ∑m,i((h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x⁽ⁱ⁾_n)

#### Features Scaling
- if multiple features, scale values to improve speed of gradient descent 
- scale them to get -1 ≤ x_i ≤ 1 (with x_i = value in dataset)
- Mean normalization :
    - replace x_i with x_i - μ_i (with μ_i = average value of x)
    - replace x_i with (x_i - μ_i)/s_i (with s_i = range (max - min))

#### Learning Rate
- debbuging :
    - plot j(θ) / Number of iterations
    - if decrease OK
    - if not, use a smaller α
    - /!\ if too small : slow convergence
- choosing :
    - α = 0.001
    - do it again with α = 3α 

#### Features and Polynomial Regression
- Features can be modify if data are appropriate:
    - in h_θ(x) = θ₀ + (θ₁ * x₁) + (θ₂ *  x₂) where x₁:L and x₂:l
    - in h_θ(x) = θ₀ + (θ₁ * (x₁ *  x₂))
- Polynomial regression :
    - for h_θ(x) = θ₀ + (θ₁ * x) ; the line will be straigth ( / )
    - for h_θ(x) = θ₀ + (θ₁ * x) + (θ₂ * √x) ; the line will be curved ( / )
    - for h_θ(x) = θ₀ + θ₁ * x + θ₂ * x² ; the line will be curved ( /\ )
    - for h_θ(x) = θ₀ + (θ₁ * x) + (θ₂ * x²) + (θ₃ * x³) ; the line will be curved ( /\/ )

#### Logistic Regression
- reminder : 
    - in classification problems -> discrete valued output (eg: true/false answers)
    - y ∈ {0, 1} where 0 is the -negative class- and 1 the -positive class-
    - mutliclass classification problems : y ∈ {0, 1, 2, ..., n}
    - don't use linear regression in classification problems
- Hypothesis :
    - h_θ(x) = g(θ^T . x)
    - g(z) = 1 / (1 + e^(-z))
    - h_θ(x) = 1 / (1 + e^(-θ^T.x))
    - h_θ(x) is now the probability that for x, y is the positive class
- Decision boundary :
    - g(z) = 0.5 if z = 0
    - θ^T.x > 0 -> h_θ(x) = 1
    - θ^T.x < 0 -> h_θ(x) = 0
    - line that separate y=1 from y=0
    - where h_θ(x) == 0.5
- Cost function :
    - J(θ) = (1/m) * ∑m,i(cost(h_θ(x⁽ⁱ⁾), y⁽ⁱ⁾))
    - cost(h_θ(x), y) = 1/2 * (h_θ(x) - y)² ❌ (non-convex function)
    - cost(h_θ(x), y) = 
        - for y = 1 : -log(h_θ(x)) ✅ (convex function)
        - for y = 0 : -log(1 - h_θ(x)) ✅ (convex function)
    - cost(h_θ(x), y) = (-y * log(h_θ(x)) - (1 - y) * log(1 - h_θ(x)) 
    - J(θ) = (-1/m) * ∑m,i((-y * log(h_θ(x)) - (1 - y) * log(1 - h_θ(x)))
- Gradient descent :
    - θ_j := θ_j - α(∂/∂*θ_j) * J(θ)
    - θ_j := θ_j - α * ∑m,i((h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x⁽ⁱ⁾_j)
- Multiclass (One-vs-all):
    - y ∈ {0, 1, 2}
    - y ∈ {(0, 1), 2} / y ∈ {(2, 0), 1} / y ∈ {(1, 2), 0}
    - h⁽ⁱ⁾_θ(x) where i ∈ y
    - to make a predict : MAX(h⁽ⁱ⁾_θ(x))
