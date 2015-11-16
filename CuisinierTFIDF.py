# -*- coding: utf-8 -*-

import math

from Cuisinier import ClassifiedRecipe, Cuisinier

"""
Uses ingredients to categorize the cuisine of a recipe via the TF-IDF algorithm
@author: Alan Kha
"""


class CuisinierTFIDF(Cuisinier):
    def __init__(self):
        super().__init__()
        self.tfidfScores = {}  # <ingredient, <cuisine, tf-idf>>

    def getAlgorithmType(self):
        return "TFIDF"

    def preprocess(self):
        super().preprocess()
        self.tfidfScores = self.calculateTFIDF()

    """
    Calculates the TF-IDF scores for each cuisine per ingredient.
    https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    @return         <ingredient, <cuisine, tf-idf>>
    """
    def calculateTFIDF(self):
        tfidfScores = {}  # <ingredient, <cuisine, tf-idf>>

        for cuisine in self.cuisineMatrix.keys():
            maxFreq = max(self.cuisineMatrix[cuisine].values())
            for ingredient, iFreq in self.cuisineMatrix[cuisine].items():
                tf = 0.5 + 0.5 * iFreq / maxFreq
                idf = math.log(len(self.cuisineMatrix) /
                               (1 + len(self.ingredientMatrix[ingredient])))

                # Add to matrix
                if ingredient not in tfidfScores:
                    tfidfScores[ingredient] = {}
                tfidfScores[ingredient][cuisine] = tf * idf

        return tfidfScores

    """
    Classifies a Recipe using TF-IDF scores. Each ingredient in the Recipe
    contributes its TF-IDF score for each cuisine, and the cuisine with the
    highest sum is selected.
    @param  Recipe  Recipe to be classified
    @return         ClassifiedRecipe
    """
    def classify(self, recipe):
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
