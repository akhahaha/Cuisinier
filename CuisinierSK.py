# -*- coding: utf-8 -*-

import sklearn
from Cuisinier import ClassifiedRecipe, Cuisinier

"""
Uses ingredients to categorize the cuisine of a recipe using sklearn CLFs.
@author: Alan Kha
"""


class CuisinierSK(Cuisinier):
    def __init__(self, clf):
        super().__init__()
        self.setCLF(clf)

    def getAlgorithmType(self):
        return self.clf.__class__.__name__

    def preprocess(self):
        super().preprocess()
        self.fitSVC()

    def setCLF(self, clf):
        if not isinstance(clf, sklearn):
            raise TypeError("CuisinierSK.setCLF requires a sklearn CLF")
        self.clf = clf
        super.preprocessed = False

    def getCLF(self):
        return self.clf

    def fitSVC(self):
        # One-hot encode recipe samples
        samples = [[1 if ingredient in recipe.ingredients else 0 for ingredient
                    in self.ingredientMatrix.keys()]
                   for recipe in self.recipes.values()]
        labels = [recipe.cuisine for recipe in self.recipes.values()]
        self.clf.fit(samples, labels)

    def classify(self, recipe):
        return ClassifiedRecipe(recipe.id,
                                self.clf.predict([recipe.ingredients]),
                                recipe.ingreidents)
