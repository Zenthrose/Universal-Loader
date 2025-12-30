#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QtGui/QVulkanWindow>
#include <vulkan/vulkan.h>

class VulkanWindow : public QVulkanWindow {
public:
    VkClearColorValue clearColor = {{0.0f, 0.0f, 1.0f, 1.0f}}; // blue

protected:
    void startNextFrame() override {
        VkClearValue clearValues[1];
        clearValues[0].color = clearColor;

        VkRenderPassBeginInfo rpBeginInfo = {};
        rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBeginInfo.renderPass = defaultRenderPass();
        rpBeginInfo.framebuffer = currentFramebuffer();
        rpBeginInfo.renderArea.extent.width = swapChainImageSize().width();
        rpBeginInfo.renderArea.extent.height = swapChainImageSize().height();
        rpBeginInfo.clearValueCount = 1;
        rpBeginInfo.pClearValues = clearValues;

        VkCommandBuffer cmdBuf = currentCommandBuffer();
        vkCmdBeginRenderPass(cmdBuf, &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdEndRenderPass(cmdBuf);

        frameReady();
        requestUpdate();
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    VulkanWindow vulkanWindow;

    QWidget widget;
    QVBoxLayout *layout = new QVBoxLayout(&widget);
    layout->addWidget(QWidget::createWindowContainer(&vulkanWindow));

    QPushButton *button = new QPushButton("Hello Vulkan");
    layout->addWidget(button);

    QObject::connect(button, &QPushButton::clicked, [&]() {
        // Change clear color to red
        vulkanWindow.clearColor.float32[0] = 1.0f;
        vulkanWindow.clearColor.float32[1] = 0.0f;
        vulkanWindow.clearColor.float32[2] = 0.0f;
        vulkanWindow.requestUpdate();
    });

    widget.setWindowTitle("Vulkan GUI");
    widget.show();

    return app.exec();
}