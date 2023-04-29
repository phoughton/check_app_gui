from decouple import config
import azure.ai.vision as sdk

service_options = sdk.VisionServiceOptions(config("VISION_ENDPOINT"),
                                           config("VISION_KEY"))

vision_source = sdk.VisionSource(
    url="https://cdn.britannica.com/23/194523-050-E6C02DBE/selection-American-playing-cards-jack-queen-ace.jpg")
# https://media.istockphoto.com/id/160042754/photo/royal-flush.jpg?s=612x612&w=0&k=20&c=vBD-6SsV6vubTRPuJTbea8TGElJssc0lHI0MOHA9scU=")
# https://i.guim.co.uk/img/media/d0d65e8cc17a815fe1fd955ae20ff47c40c58988/0_0_3000_2000/master/3000.jpg?width=620&quality=85&dpr=1&s=none")
# https://ichef.bbci.co.uk/news/976/cpsprodpb/1B21/production/_129254960_microsoftteams-image.png")
# https://learn.microsoft.com/azure/cognitive-services/computer-vision/media/quickstarts/presentation.png")

analysis_options = sdk.ImageAnalysisOptions()

analysis_options.features = (
    sdk.ImageAnalysisFeature.CAPTION |
    sdk.ImageAnalysisFeature.TEXT |
    sdk.ImageAnalysisFeature.OBJECTS
)

analysis_options.language = "en"

analysis_options.gender_neutral_caption = True

image_analyzer = sdk.ImageAnalyzer(
    service_options, vision_source, analysis_options)

result = image_analyzer.analyze()

if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:

    if result.caption is not None:
        print(" Caption:")
        print("   '{}', Confidence {:.4f}".format(
            result.caption.content, result.caption.confidence))

    if result.objects is not None:
        print(" Objects:")
        for an_object in result.objects:
            print("   '{}', {} Confidence: {:.4f}".format(
                an_object.name, an_object.bounding_box, an_object.confidence))

    if result.text is not None:
        print(" Text:")
        for line in result.text.lines:
            points_string = "{" + ", ".join([str(int(point)) for point in line.bounding_polygon]) + "}"
            print("   Line: '{}', Bounding polygon {}".format(line.content, points_string))
            for word in line.words:
                points_string = "{" + ", ".join([str(int(point)) for point in word.bounding_polygon]) + "}"
                print("     Word: '{}', Bounding polygon {}, Confidence {:.4f}"
                      .format(word.content, points_string, word.confidence))

    result_details = sdk.ImageAnalysisResultDetails.from_result(result)
    print(" Result details:")
    print("   Image ID: {}".format(result_details.image_id))
    print("   Result ID: {}".format(result_details.result_id))
    print("   Connection URL: {}".format(result_details.connection_url))
    print("   JSON result: {}".format(result_details.json_result))


elif result.reason == sdk.ImageAnalysisResultReason.ERROR:

    error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
    print(" Analysis failed.")
    print("   Error reason: {}".format(error_details.reason))
    print("   Error code: {}".format(error_details.error_code))
    print("   Error message: {}".format(error_details.message))

