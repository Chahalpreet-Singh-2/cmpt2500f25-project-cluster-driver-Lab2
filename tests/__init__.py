def test_function_name(): # 1. Name starts with 'test_'
 """Docstring explaining what we test.""" # 2. Documentation

 # 3. Arrange: Set up test data
 X = np.array([[1, 2], [3, 4]])
 y = np.array([0, 1])

 # 4. Act: Execute the code being tested
 model = train_model(X, y)

 # 5. Assert: Verify the results
 assert model is not None
 assert hasattr(model, 'predict')
