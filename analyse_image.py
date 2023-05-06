from decouple import config
import azure.ai.vision as sdk
import cv2
import os


# This function will draw a rectangle around the object and label it
def label_objects(image_source, image_out_filename, objects_in_image):

    image_to_draw = cv2.imread(image_source)

    image_width = image_to_draw.shape[1]
    image_height = image_to_draw.shape[0]

    label_text_scale = 0.001 * image_height

    print(f"Image width: {image_width}, height: {image_height}")

    # Loop through objects 
    for one_object in objects_in_image:
        print(one_object.name)

        # Assign bounding box dimensions to objects representing 
        # x an y positions as well as width and height
        x = one_object.bounding_box.x
        y = one_object.bounding_box.y
        w = one_object.bounding_box.w
        h = one_object.bounding_box.h
        print(x, y, w, h)

        cv2.rectangle(image_to_draw, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # write a text label on the image using cv2
        cv2.putText(image_to_draw, one_object.name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, label_text_scale, (0, 255, 0), 2)

        # save the image to file
        print(f"Saving image to {image_out_filename}")
    cv2.imwrite(image_out_filename, image_to_draw)


service_options = sdk.VisionServiceOptions(config("VISION_ENDPOINT"),
                                           config("VISION_KEY"))

# Image base folder location
IMAGE_FOLDER = "images/"

OUTPUT_FOLDER = "output/"

# Automatically Print a list of images in the images folder to analyse
print("Select an image to analyse:")
for index, filename in enumerate(os.listdir(IMAGE_FOLDER)):
    print(f"{index} - {filename}")

image_index = int(input("Enter the number of the image to analyse: "))
source_image_filename = os.listdir(IMAGE_FOLDER)[image_index]

source_image = IMAGE_FOLDER + source_image_filename

vision_source = sdk.VisionSource(
    filename=source_image)

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
            points_string = "{" + ", ".join([str(int(point))
                                            for point in line.bounding_polygon]) + "}"
            print("   Line: '{}', Bounding polygon {}".format(
                line.content, points_string))
            for word in line.words:
                points_string = "{" + ", ".join([str(int(point))
                                                for point in word.bounding_polygon]) + "}"
                print("     Word: '{}', Bounding polygon {}, Confidence {:.4f}"
                      .format(word.content, points_string, word.confidence))

    # label the objects in the image
    label_objects(source_image, OUTPUT_FOLDER +
                  "objects_in_" + source_image_filename, result.objects)

elif result.reason == sdk.ImageAnalysisResultReason.ERROR:

    error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
    print(" Analysis failed.")
    print("   Error reason: {}".format(error_details.reason))
    print("   Error code: {}".format(error_details.error_code))
    print("   Error message: {}".format(error_details.message))

