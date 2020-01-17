package info.benjaminhill.superdeep

import ai.djl.modality.cv.ImageVisualization
import ai.djl.mxnet.zoo.MxModelZoo
import ai.djl.mxnet.zoo.nlp.qa.QAInput
import ai.djl.training.util.ProgressBar
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

private fun detectObjects(imageFile: File): BufferedImage {
    require(imageFile.canRead())
    val img = ImageIO.read(imageFile)
    val criteria = mapOf(
        "size" to "512",
        "backbone" to "resnet50",
        "flavor" to "v1",
        "dataset" to "voc"
    )
    MxModelZoo.SSD.loadModel(criteria, ProgressBar()).use { model ->
        model.newPredictor().use { predictor ->
            val detection = predictor.predict(img)
            val newImage = BufferedImage(img.width, img.height, BufferedImage.TYPE_INT_ARGB)
            newImage.createGraphics().let { g ->
                g.drawImage(img, 0, 0, null)
                g.dispose()
            }
            ImageVisualization.drawBoundingBoxes(newImage, detection)
            return newImage
        }
    }
}


private fun predictAnswer() {
    val paragraph = ("""BBC Japan was a general entertainment Channel.
Which operated between December 2004 and April 2006.
It ceased operations after its Japanese distributor folded.""")
    val criteria = mapOf(
        "backbone" to "bert",
        "dataset" to "book_corpus_wiki_en_uncased"
    )
    arrayOf(
        "When did BBC Japan start broadcasting?",
        "When did BBC Japan stop broadcasting?"
    ).forEach { question ->
        val input = QAInput(question, paragraph, 384)
        println("Paragraph: ${input.paragraph}")
        println("Question: ${input.question}")
        MxModelZoo.BERT_QA.loadModel(criteria, ProgressBar()).use { model ->
            model.newPredictor().use { predictor ->
                println("Answer: ${predictor.predict(input)}")
            }
        }
    }
}


fun main() {
    val imageFile = File("src/main/resources/dog_bike_car.jpg")
    ImageIO.write(detectObjects(imageFile), "png", File("output_${imageFile.nameWithoutExtension}.png"))

    predictAnswer()
}

