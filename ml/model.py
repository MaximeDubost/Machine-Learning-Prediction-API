from utils.utils import DataHandler, FeatureRecipe, FeatureExtractor

def DataManager(d:DataHandler=None, fr: FeatureRecipe=None, fe:FeatureExtractor=None):
    """
        Fonction qui lie les 3 premières classes de la pipeline et qui return FeatureExtractor.split(0.1)
    """

    data_handler = DataHandler()
    df = data_handler.get_process_data()

    feature_recipe = FeatureRecipe(df)
    feature_recipe.prepare_data(0.3)

    feature_extractor = FeatureExtractor(data_handler.df, list(['local_price', 'latitude', 'longitude', 'pricing_weekly_factor', 'pricing_monthly_factor', 'beds', 'bedrooms', 'bathrooms']))
    feature_extractor.extract()
    return feature_extractor.split_data(0.3, 42, 'local_price')