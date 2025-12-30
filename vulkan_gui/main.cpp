#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QTextEdit>
#include <QLineEdit>
#include <QComboBox>
#include <QSlider>
#include <QCheckBox>
#include <QLabel>
#include <QFileDialog>
#include <QInputDialog>
#include <QProcess>
#include <QThread>
#include <QTimer>
#include <QVulkanWindow>
#include <QVulkanWindowRenderer>
#include <QVulkanDeviceFunctions>
#include <vulkan/vulkan.h>
#include <QObject>
#include <QMessageBox>
#include <stdexcept>

#include "../src/inference/inference_engine.h"
#include "../src/inference/bpe_tokenizer.h"

using namespace inference;

class VulkanRenderer : public QVulkanWindowRenderer {
public:
    VulkanRenderer(QVulkanWindow *w) : m_window(w) {}

    void initResources() {
        // Initialize resources if needed
    }

    void startNextFrame() {
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

    void releaseResources() {
        // Release resources if needed
    }

    VkClearColorValue m_clearColor = {{0.0f, 0.0f, 1.0f, 1.0f}}; // blue

private:
    QVulkanWindow *m_window;
};

class VulkanWindow : public QVulkanWindow {
public:
    VulkanRenderer *m_renderer;

    QVulkanWindowRenderer *createRenderer() {
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

class GenerationThread : public QThread {
public:
    GenerationThread(InferenceEngine* eng, const QString& pr) : engine(eng), prompt(pr) {}

    void run() override {
        response = QString::fromStdString(engine->generate(prompt.toStdString(), 100));
    }

    QString response;

private:
    InferenceEngine* engine;
    QString prompt;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // Main widget
    QWidget mainWidget;
    mainWidget.setWindowTitle("VulkanGGUF Chat");
    mainWidget.resize(1200, 800);

    QHBoxLayout *mainLayout = new QHBoxLayout(&mainWidget);

    // Sidebar
    QWidget *sidebar = new QWidget;
    sidebar->setFixedWidth(250);
    QVBoxLayout *sidebarLayout = new QVBoxLayout(sidebar);

    QLabel *loraLabel = new QLabel("LoRA Adapter:");
    QComboBox *loraCombo = new QComboBox;
    loraCombo->addItem("None");

    QLabel *tempLabel = new QLabel("Temperature: 1.0");
    QSlider *tempSlider = new QSlider(Qt::Horizontal);
    tempSlider->setRange(0, 200);
    tempSlider->setValue(100);

    QLabel *topPLabel = new QLabel("Top P: 1.0");
    QSlider *topPSlider = new QSlider(Qt::Horizontal);
    topPSlider->setRange(0, 100);
    topPSlider->setValue(100);

    QCheckBox *speculativeCheck = new QCheckBox("Speculative Decoding");

    QLabel *statsLabel = new QLabel("Stats: Loading...");

    sidebarLayout->addWidget(loraLabel);
    sidebarLayout->addWidget(loraCombo);
    sidebarLayout->addWidget(tempLabel);
    sidebarLayout->addWidget(tempSlider);
    sidebarLayout->addWidget(topPLabel);
    sidebarLayout->addWidget(topPSlider);
    sidebarLayout->addWidget(speculativeCheck);
    sidebarLayout->addWidget(statsLabel);
    sidebarLayout->addStretch();

    // Vulkan debug panel
    QWidget *vulkanContainer = new QWidget;
    vulkanContainer->setFixedHeight(200);
    QVBoxLayout *vulkanLayout = new QVBoxLayout(vulkanContainer);
    VulkanWindow *vulkanWindow = nullptr;
    try {
        vulkanWindow = new VulkanWindow;
    } catch (std::exception& e) {
        QMessageBox::warning(nullptr, "Vulkan Error", QString("Vulkan not available: %1").arg(e.what()));
    }
    if (vulkanWindow) {
        vulkanLayout->addWidget(QWidget::createWindowContainer(vulkanWindow));
    } else {
        QLabel *vulkanLabel = new QLabel("Vulkan not available");
        vulkanLayout->addWidget(vulkanLabel);
    }
    QPushButton *debugButton = new QPushButton("Change Clear Color");
    vulkanLayout->addWidget(debugButton);
    if (vulkanWindow) {
        QObject::connect(debugButton, &QPushButton::clicked, [vulkanWindow]() {
            vulkanWindow->setClearColor(1.0f, 0.0f, 0.0f);
        });
    } else {
        debugButton->setEnabled(false);
    }
    sidebarLayout->addWidget(new QLabel("Vulkan Debug:"));
    sidebarLayout->addWidget(vulkanContainer);

    // Main area
    QWidget *mainArea = new QWidget;
    QVBoxLayout *mainAreaLayout = new QVBoxLayout(mainArea);

    // Top buttons
    QHBoxLayout *topLayout = new QHBoxLayout;
    QPushButton *loadLocalButton = new QPushButton("Load Local GGUF");
    QPushButton *downloadHFButton = new QPushButton("Download from HuggingFace");
    topLayout->addWidget(loadLocalButton);
    topLayout->addWidget(downloadHFButton);
    topLayout->addStretch();
    mainAreaLayout->addLayout(topLayout);

    // Chat area
    QTextEdit *chatEdit = new QTextEdit;
    chatEdit->setReadOnly(true);
    chatEdit->setStyleSheet("QTextEdit { background-color: #f5f5f5; font-family: Arial; font-size: 12px; }");
    mainAreaLayout->addWidget(chatEdit);

    // Input area
    QHBoxLayout *inputLayout = new QHBoxLayout;
    QTextEdit *inputEdit = new QTextEdit;
    inputEdit->setMaximumHeight(80);
    inputEdit->setPlaceholderText("Enter your prompt...");
    QPushButton *sendButton = new QPushButton("Send");
    sendButton->setEnabled(false);
    inputLayout->addWidget(inputEdit);
    inputLayout->addWidget(sendButton);
    mainAreaLayout->addLayout(inputLayout);

    // Add to main layout
    mainLayout->addWidget(sidebar);
    mainLayout->addWidget(mainArea);

    QString chatHistory;

    InferenceEngine *engine = nullptr;
    BPETokenizer *tokenizer = nullptr;

    // Stats timer
    QTimer *statsTimer = new QTimer;
    statsTimer->start(1000);
    QObject::connect(statsTimer, &QTimer::timeout, [&]() {
        if (engine) {
            statsLabel->setText(QString("Tokens/s: %1, Memory: %2 MB").arg(engine->get_throughput()).arg(engine->get_model_size_bytes() / 1024 / 1024));
        } else {
            statsLabel->setText("Stats: Loading...");
        }
    });

    // Connect load local
    QObject::connect(loadLocalButton, &QPushButton::clicked, [&]() {
        QString fileName = QFileDialog::getOpenFileName(&mainWidget, "Load GGUF Model", "", "GGUF Files (*.gguf)");
        if (!fileName.isEmpty()) {
            delete engine;
            engine = new InferenceEngine();
            InferenceConfig config = {1024, 4, 1024LL*1024*1024, 512LL*1024*1024, 2048, 2, BackendType::GPU, true};
            if (!engine->initialize(config)) {
                QMessageBox::critical(&mainWidget, "Error", "Failed to initialize engine");
                delete engine;
                engine = nullptr;
                return;
            }
            if (!engine->load_model(fileName.toStdString())) {
                QMessageBox::critical(&mainWidget, "Error", "Failed to load model");
                delete engine;
                engine = nullptr;
                return;
            }
            delete tokenizer;
            tokenizer = new BPETokenizer();
            // tokenizer->load_from_file(fileName.toStdString() + ".tokenizer.json");
            sendButton->setEnabled(true);
            statsLabel->setText("Model loaded");
        }
    });

    // Connect download HF
    QObject::connect(downloadHFButton, &QPushButton::clicked, [&]() {
        bool ok;
        QString repo = QInputDialog::getText(&mainWidget, "Download from HuggingFace", "Repo ID (e.g., microsoft/DialoGPT-medium):", QLineEdit::Normal, "", &ok);
        if (ok && !repo.isEmpty()) {
            QProcess *process = new QProcess;
            process->start("huggingface-cli", QStringList() << "download" << repo << "model.gguf" << "--local-dir" << ".");
            statsLabel->setText("Downloading...");
            QObject::connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), [&](int exitCode) {
                if (exitCode == 0) {
                    QString fileName = "model.gguf";
                    delete engine;
                    engine = new InferenceEngine();
                    InferenceConfig config = {1024, 4, 1024LL*1024*1024, 512LL*1024*1024, 2048, 2, BackendType::GPU, true};
                    if (!engine->initialize(config)) {
                        QMessageBox::critical(nullptr, "Error", "Failed to initialize engine");
                        delete engine;
                        engine = nullptr;
                        statsLabel->setText("Initialization failed");
                        return;
                    }
                    if (!engine->load_model(fileName.toStdString())) {
                        QMessageBox::critical(nullptr, "Error", "Failed to load model");
                        delete engine;
                        engine = nullptr;
                        statsLabel->setText("Load failed");
                        return;
                    }
                    delete tokenizer;
                    tokenizer = new BPETokenizer();
                    // tokenizer->load_from_file(fileName.toStdString() + ".tokenizer.json");
                    sendButton->setEnabled(true);
                    statsLabel->setText("Downloaded and loaded");
                } else {
                    statsLabel->setText("Download failed");
                }
            });
        }
    });

    // Connect send
    QObject::connect(sendButton, &QPushButton::clicked, [&]() {
        QString prompt = inputEdit->toPlainText();
        if (prompt.isEmpty() || !engine) return;
        inputEdit->clear();
        chatHistory += "<div style='text-align: right; background-color: #007bff; color: white; padding: 5px; border-radius: 5px; margin: 5px;'>" + prompt.toHtmlEscaped() + "</div>";
        chatEdit->setHtml(chatHistory);
        chatHistory += "<div style='text-align: left; background-color: #28a745; color: white; padding: 5px; border-radius: 5px; margin: 5px;'>";
        GenerationThread *thread = new GenerationThread(engine, prompt);
        QObject::connect(thread, &QThread::finished, [&, thread]() {
            chatHistory += thread->response.toHtmlEscaped() + "</div>";
            chatEdit->setHtml(chatHistory);
            thread->deleteLater();
        });
        thread->start();
    });

    // Connect sliders
    QObject::connect(tempSlider, &QSlider::valueChanged, [&](int value) {
        tempLabel->setText(QString("Temperature: %1").arg(value / 100.0));
    });

    QObject::connect(topPSlider, &QSlider::valueChanged, [&](int value) {
        topPLabel->setText(QString("Top P: %1").arg(value / 100.0));
    });

    mainWidget.show();
    return app.exec();
}