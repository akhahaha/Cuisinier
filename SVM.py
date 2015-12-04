import math
from Cuisinier import Recipe, ClassifiedRecipe, Cuisinier
from sklearn import svm,linear_model,decomposition
import logging

"""
Uses ingredients to categorize the cuisine of a recipe via the
SVM algorithm.
@author: Sravani Kamisetty
Training Set: X - One count encoded recipes; Y - Cuisine
Testing Set: X - One count encoded recipe;Y - predict the cuisine
Size of training set X = #uniqueCusines * #uniqueIngredients
"""


class SVM(Cuisinier):
    def __init__(self):
        super().__init__()
        self.XTrain = []
        self.YTrain = []
        self.XTest = []
        self.YTest = []

    def preprocess(self):
        super().preprocess()
        self.constructMatrix()

    def getIndex(self,array,key):
        j = 0
        for i in array:
            if(i == key):
                return j
            j = j+1;
        return -1

    def convertToDict(self, array):
        dict  = {}
        j = 1
        for i in array:
            dict[i] = j;
            j = j+1;
        return dict

    """
    Constructs the training matrix X - one count encoded recepies
    @return
    """
    def constructMatrix(self):

        X = []
        Y = []
        cuisines = list(self.cuisineMatrix.keys())
        ingredients = list(self.ingredientMatrix.keys())
        cuisinesDict = self.convertToDict(cuisines)
        ingredientsDict = self.convertToDict(ingredients)

        logging.info("Unique Cuisises: " + cuisinesDict)
        #print(ingredientsDict)

        x = []
        x1 = []
        for i in range(0,6734):
            x.append(0)
            x1.append(0)

        count = 0
        logging.info(self.recipes.__len__())
        for recipeId in self.recipes:
            recipe = self.recipes[recipeId]
            Y.append(cuisinesDict.get(recipe.cuisine))
            for ingredient in recipe.ingredients:
                index = ingredientsDict.get(ingredient)
                if index >=0 and index< ingredients.__len__():
                    x[index] = 1;
                else:
                    logging.info("Ingredient not present in recipe" + ingredient + ":" + str(index) + ":" + str(recipe))
            X.append(x)
            x = x1

        self.XTrain = X
        self.YTrain = Y

        X = []
        Y = []
        for recipeId in self.recipesTest:
            recipe = self.recipesTest[recipeId]
            Y.append(cuisinesDict.get(recipe.cuisine))
            for ingredient in recipe.ingredients:
                index = ingredientsDict.get(ingredient)
                if index >=0 and index< ingredients.__len__():
                    x[index] = 1;
                else:
                    logging.info("Ingredient not present in recipe" + ingredient + ":" + str(index) + ":" + str(recipe))
            X.append(x)
            x = x1

        self.XTest = X
        self.YTest = Y

    """
    Computes the accuracy of model
    @params Y Cuisine
    YPredicted Predicted Cuisine
    @return accuracy

    """
    def computeAccuracy(self,Y,YPredicted):
        count = 0
        for i in range(0,len(Y)):
            if Y[i] == YPredicted[i]:
                count = count + 1

        return (count/len(Y) * 100)
    """
    Constructs a model using the training dataset.
    """
    def constructModel(self):

        '''
        Dimension reduction using PCA
        '''
        pca = decomposition.PCA(n_components=100)
        pca.fit(self.XTrain)
        self.XTrainReduced = pca.transform(self.XTrain)
        self.XTestReduced = pca.transform(self.XTest)

        '''
        Logistic Regression
        '''
        self.clf = svm.SVC()
        print("Model construction begins")
        self.clf.fit(self.XTrainReduced,self.YTrain)
        print("Model constructed")
        YPredicted = self.clf.predict(self.XTrainReduced)
        print("Model test")
        YTestPredicted = self.clf.predict(self.XTest);

        '''
        SVM
        self.clf = linear_model.LogisticRegression(C=1e5)
        print("Model construction begins")
        self.clf.fit(self.XTrain,self.YTrain)
        print("Model constructed")
        YPredicted = self.clf.predict(self.XTrain)
        print("Model predictions done")
        '''
        accuracyTrain = self.computeAccuracy(self.YTrain,YPredicted)
        print("Accuracy of training:" + str(accuracyTrain));
        accuracyTest = self.computeAccuracy(self.YTrain,YPredicted)
        print("Accuracy of training:" + str(accuracyTest));

    """
    Classifies a Recipe using one hot encded matrix. Each ingredient in the Recipe
    contriutes its TF-IDF score for each cuisine, and the cuisine with the
    highest sum is selected.
    @param  Recipe  Recipe to be classified
    @return         ClassifiedRecipe
    """
    def classify(self, recipe):

        '''
        cuisineScores = {}
        for ingredient in recipe.ingredients:
            if ingredient in self.tfidfScores:
                for cuisine, score in self.tfidfScores[ingredient].items():
                    if cuisine not in cuisineScores:
                        cuisineScores[cuisine] = 0
                    cuisineScores[cuisine] += score

        # Select highest scoring cuisine
        bestCuisine = max(cuisineScores.keys(),
                          key=(lambda key: cuisineScores[key]))

        return ClassifiedRecipe(recipe.id, bestCuisine, recipe.ingredients)
        '''
