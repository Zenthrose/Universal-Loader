#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QVulkanWindow>
#include <QVulkanDeviceFunctions>
#include <vulkan/vulkan.h>

class VulkanRenderer : public QVulkanWindowRenderer {
public:
    VulkanRenderer(QVulkanWindow *w) : m_window(w) {}

    void initResources() override {
        // Initialize resources if needed
    }

    void startNextFrame() override {
        QVulkanDeviceFunctions *df = m_window->vulkanInstance()->deviceFunctions(m_window->device());

        VkClearValue clearValues[1];
        clearValues[0].color = m_clearColor;

        VkRenderPassBeginInfo rpBeginInfo = {};
        rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBeginInfo.renderPass = m_window->defaultRenderPass();
        rpBeginInfo.framebuffer = m_window->currentFramebuffer();
        rpBeginInfo.renderArea.extent.width = m_window->swapChainImageSize().width();
        rpBeginInfo.renderArea.extent.height = m_window->swapChainImageSize().height();
        rpBeginInfo.clearValueCount = 1;
        rpBeginInfo.pClearValues = clearValues;

        VkCommandBuffer cmdBuf = m_window->currentCommandBuffer();
        df->vkCmdBeginRenderPass(cmdBuf, &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        df->vkCmdEndRenderPass(cmdBuf);

        m_window->frameReady();
        m_window->requestUpdate();
    }

    void releaseResources() override {
        // Release resources if needed
    }

    VkClearColorValue m_clearColor = {{0.0f, 0.0f, 1.0f, 1.0f}}; // blue

private:
    QVulkanWindow *m_window;
};

class VulkanWindow : public QVulkanWindow {
public:
    VulkanRenderer *m_renderer;

    QVulkanWindowRenderer *createRenderer() override {
        m_renderer = new VulkanRenderer(this);
        return m_renderer;
    }

    void setClearColor(float r, float g, float b) {
        m_renderer->m_clearColor.float32[0] = r;
        m_renderer->m_clearColor.float32[1] = g;
        m_renderer->m_clearColor.float32[2] = b;
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
        vulkanWindow.setClearColor(1.0f, 0.0f, 0.0f);
    });

    widget.setWindowTitle("Vulkan GUI");
    widget.show();

    return app.exec();
}