import os
from decouple import config
import azure.ai.vision as sdk
import cv2


# This function will draw a rectangle around the text and label it
def label_text(image_source, image_out_filename, text_in_image):

    image_to_draw = cv2.imread(image_source)

    image_width = image_to_draw.shape[1]
    image_height = image_to_draw.shape[0]

    print(f"Image width: {image_width}, height: {image_height}")

    for a_line in text_in_image.lines:

        x = int(a_line.bounding_polygon[0])
        y = int(a_line.bounding_polygon[1])
        w = int(a_line.bounding_polygon[2]) - int(a_line.bounding_polygon[0])
        h = int(a_line.bounding_polygon[5]) - int(a_line.bounding_polygon[1])

        cv2.rectangle(image_to_draw, (x, y), (x + w, y + h), (255, 0, 0), 3)

    print(f"Saving image to {image_out_filename}")
    cv2.imwrite(image_out_filename, image_to_draw)


# This function will draw a rectangle around the object and label it
def label_objects(image_source, image_out_filename, objects_in_image):

    image_to_draw = cv2.imread(image_source)
    image_height = image_to_draw.shape[0]

    label_text_scale = 0.001 * image_height

    for one_object in objects_in_image:
        print(one_object.name)

        x = one_object.bounding_box.x
        y = one_object.bounding_box.y
        w = one_object.bounding_box.w
        h = one_object.bounding_box.h

        cv2.rectangle(image_to_draw, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(image_to_draw, one_object.name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, label_text_scale, (0, 255, 0), 2)

    print(f"Saving image to {image_out_filename}")
    cv2.imwrite(image_out_filename, image_to_draw)


# Function actually calls the cognitive services api
def analyse_image(image_source):

    service_options = sdk.VisionServiceOptions(config("VISION_ENDPOINT"),
                                               config("VISION_KEY"))

    vision_source = sdk.VisionSource(
        filename=image_source)

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

    return image_analyzer.analyze()


def main():
    # Image base folder location
    IMAGE_FOLDER = "images/"
    OUTPUT_FOLDER = "output/"

    # Automatically Print a list of images in the images folder to analyse
    print("Select an image to analyse:")
    for index, filename in enumerate(os.listdir(IMAGE_FOLDER)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(f"{index} - {filename}")

    image_index = int(input("Enter the number of the image to analyse: "))
    source_image_filename = os.listdir(IMAGE_FOLDER)[image_index]

    source_image = IMAGE_FOLDER + source_image_filename

    result = analyse_image(source_image)

    if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:

        if result.caption is not None:
            print(" Caption:")
            print(f" {result.caption.content}," +
                  "Confidence {result.caption.confidence}")

        if result.objects is not None:
            print(" Objects:")
            for an_object in result.objects:
                print(f" '{an_object.name}', Conf: {an_object.confidence}")

        if result.text is not None:
            print(" Text:")
            for line in result.text.lines:
                points_string = "{" + str(line.bounding_polygon) + "}"
                print(f" Line: '{line.content}', Bounding: {points_string}")
                for word in line.words:
                    print(f" Word: '{word.content}', Conf: {word.confidence}")

        label_objects(source_image, OUTPUT_FOLDER +
                      "objects_in_" + source_image_filename, result.objects)

        label_text(source_image, OUTPUT_FOLDER +
                   "text_in_" + source_image_filename, result.text)

    elif result.reason == sdk.ImageAnalysisResultReason.ERROR:

        error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
        print(" Analysis failed.")
        print("   Error reason: {}".format(error_details.reason))
        print("   Error code: {}".format(error_details.error_code))
        print("   Error message: {}".format(error_details.message))


if __name__ == "__main__":
    main()
