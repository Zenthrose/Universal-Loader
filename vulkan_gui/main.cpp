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
#include <QVulkanInstance>
#include <vulkan/vulkan.h>
#include <QObject>
#include <QMessageBox>
#include <QProgressBar>
#include <QDir>
#include <QDirIterator>
#include <stdexcept>
#include <QMenuBar>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QUrl>
#include <QSpinBox>
#include <QKeyEvent>
#include <QFile>
#include <QTextStream>
#include <QDateTime>
#include <QFileInfo>

// Project headers
#include "../src/inference/inference_engine.h"
#include "../src/inference/bpe_tokenizer.h"

using namespace inference;

// Global or shared engine instance
static InferenceEngine* g_engine = nullptr;

// Debug logger to catch silent crashes on Windows
void debugLogHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
    QFile outFile("debug_log.txt");
    if (outFile.open(QIODevice::WriteOnly | QIODevice::Append)) {
        QTextStream ts(&outFile);
        ts << QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss.zzz ") << msg << Qt::endl;
    }
}

class ChatEventFilter : public QObject {
public:
    ChatEventFilter(QPushButton* sendBtn, QObject* parent = nullptr) 
        : QObject(parent), m_sendButton(sendBtn) {}

protected:
    bool eventFilter(QObject* obj, QEvent* event) override {
        if (event->type() == QEvent::KeyPress) {
            QKeyEvent* keyEvent = static_cast<QKeyEvent*>(event);
            if (keyEvent->key() == Qt::Key_Return && !(keyEvent->modifiers() & Qt::ShiftModifier)) {
                m_sendButton->animateClick();
                return true;
            }
        }
        return QObject::eventFilter(obj, event);
    }
private:
    QPushButton* m_sendButton;
};

class VulkanRenderer : public QVulkanWindowRenderer {
public:
    VulkanRenderer(QVulkanWindow *w) : m_window(w) {}
    void initResources() override {}
    void startNextFrame() override {
        QVulkanDeviceFunctions *df = m_window->vulkanInstance()->deviceFunctions(m_window->device());
        VkClearValue clearValues[1];
        clearValues[0].color = {{ 0.05f, 0.05f, 0.07f, 1.0f }};
        VkRenderPassBeginInfo rpBeginInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, nullptr, m_window->defaultRenderPass(), m_window->currentFramebuffer(), {{0,0}, {uint32_t(m_window->width()), uint32_t(m_window->height())}}, 1, clearValues };
        df->vkCmdBeginRenderPass(m_window->currentCommandBuffer(), &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        df->vkCmdEndRenderPass(m_window->currentCommandBuffer());
        m_window->frameReady();
        m_window->requestUpdate(); 
    }
private:
    QVulkanWindow *m_window;
};

class VulkanWindow : public QVulkanWindow {
public:
    VulkanWindow() {
        QWindow::setFlags(Qt::Window | Qt::FramelessWindowHint);
    }
    QVulkanWindowRenderer *createRenderer() override { return new VulkanRenderer(this); }
};

int main(int argc, char *argv[]) {
    qInstallMessageHandler(debugLogHandler);
    
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
    
    QApplication app(argc, argv);
    app.setProperty("platformName", "windows");

    qDebug() << "Application starting, initializing Vulkan...";

    QVulkanInstance inst;
    inst.setLayers({}); 

    if (!inst.create()) {
        qCritical() << "Vulkan Instance creation failed!";
        QMessageBox::critical(nullptr, "Vulkan Fatal Error", 
            "The application could not initialize Vulkan.\n\n"
            "Ensure your RX 580 has Adrenalin 23.x or newer drivers installed.");
        return 1;
    }
    qDebug() << "Vulkan Instance created.";

    QWidget *mainWidget = new QWidget();
    mainWidget->setWindowTitle("Vulkan GGUF Loader");
    mainWidget->resize(1280, 720);

    QHBoxLayout *mainLayout = new QHBoxLayout(mainWidget);
    QWidget *sidebar = new QWidget();
    sidebar->setFixedWidth(300);
    QVBoxLayout *sidebarLayout = new QVBoxLayout(sidebar);

    sidebarLayout->addWidget(new QLabel("Active Model"));
    QComboBox *modelCombo = new QComboBox();
    modelCombo->addItem("No Model Loaded"); 
    sidebarLayout->addWidget(modelCombo);

    QPushButton *loadModelBtn = new QPushButton("Load GGUF Model");
    sidebarLayout->addWidget(loadModelBtn);

    sidebarLayout->addSpacing(20);
    sidebarLayout->addWidget(new QLabel("Inference Settings"));
    
    QLabel *tempLabel = new QLabel("Temperature: 0.70");
    sidebarLayout->addWidget(tempLabel);
    QSlider *tempSlider = new QSlider(Qt::Horizontal);
    tempSlider->setRange(0, 200); tempSlider->setValue(70);
    sidebarLayout->addWidget(tempSlider);

    QCheckBox *gpuOffload = new QCheckBox("RX 580 Vulkan Acceleration");
    gpuOffload->setChecked(true);
    sidebarLayout->addWidget(gpuOffload);

    sidebarLayout->addStretch();
    QProgressBar *vramBar = new QProgressBar();
    vramBar->setFormat("VRAM Usage: %v MB");
    sidebarLayout->addWidget(vramBar);

    QVBoxLayout *rightLayout = new QVBoxLayout();
    
    VulkanWindow *vulkanWin = new VulkanWindow();
    vulkanWin->setVulkanInstance(&inst);
    QWidget *vulkanContainer = QWidget::createWindowContainer(vulkanWin, mainWidget);
    vulkanContainer->setMinimumHeight(200);
    rightLayout->addWidget(vulkanContainer, 1);

    QTextEdit *chatHistory = new QTextEdit();
    chatHistory->setReadOnly(true);
    chatHistory->setStyleSheet("background-color: #121212; color: #00FFCC; font-family: 'Consolas'; border: 1px solid #333;");
    rightLayout->addWidget(chatHistory, 2);

    QHBoxLayout *inputLayout = new QHBoxLayout();
    QTextEdit *inputEdit = new QTextEdit();
    inputEdit->setFixedHeight(80);
    QPushButton *sendButton = new QPushButton("Inference");
    sendButton->setFixedWidth(120);
    sendButton->setFixedHeight(80);
    inputLayout->addWidget(inputEdit);
    inputLayout->addWidget(sendButton);
    rightLayout->addLayout(inputLayout);

    mainLayout->addWidget(sidebar);
    mainLayout->addLayout(rightLayout);

    ChatEventFilter* filter = new ChatEventFilter(sendButton, mainWidget);
    inputEdit->installEventFilter(filter);

    QObject::connect(loadModelBtn, &QPushButton::clicked, [=]() {
    QString fileName = QFileDialog::getOpenFileName(mainWidget, "Select Model", "", "GGUF Files (*.gguf)");
    if (!fileName.isEmpty()) {
        QFileInfo checkFile(fileName);
        QString modelName = checkFile.fileName();
        
        // FIX: Add comprehensive validation
        if (!checkFile.exists()) {
            chatHistory->append("<b style='color:#FF4444;'>Error:</b> File does not exist");
            return;
        }
        
        if (!checkFile.isReadable()) {
            chatHistory->append("<b style='color:#FF4444;'>Error:</b> File is not readable");
            return;
        }
        
        if (checkFile.size() < 1024) {  // FIX: Minimum GGUF size check
            chatHistory->append("<b style='color:#FF4444;'>Error:</b> File too small to be a valid GGUF");
            return;
        }
        
        // FIX: Validate GGUF magic bytes before loading
        std::ifstream test_file(fileName.toStdString(), std::ios::binary);
        char magic[4];
        test_file.read(magic, 4);
        test_file.close();
        
        if (std::string(magic, 4) != "GGUF") {
            chatHistory->append("<b style='color:#FF4444;'>Error:</b> Not a valid GGUF file (invalid magic bytes)");
            return;
        }
        
        qDebug() << "Selected File:" << fileName << "Size:" << checkFile.size() << "bytes";
        
        try {
            if (g_engine) {
                delete g_engine;
                g_engine = nullptr;
            }
            
            g_engine = new InferenceEngine();

            InferenceConfig config;
            config.max_tokens = 2048;
            config.num_threads = 4;
            config.gpu_memory_pool_mb = 2048; 
            config.gpu_cache_mb = 512;
            config.context_len = 2048; 
            config.prefetch_layers = 10;
            config.backend = gpuOffload->isChecked() ? BackendType::GPU : BackendType::CPU;
            config.enable_validation = false;

            qDebug() << "Initializing engine...";
            if (!g_engine->initialize(config)) {
                qWarning() << "GPU backend initialization failed. Attempting CPU fallback.";
                config.backend = BackendType::CPU;
                if (!g_engine->initialize(config)) {
                    chatHistory->append("<b style='color:#FF4444;'>Error:</b> Engine initialization failed on all backends");
                    return;
                }
            }

            qDebug() << "Parsing GGUF structure...";
            if (g_engine->load_model(fileName.toStdString())) {
                modelCombo->clear();
                modelCombo->addItem(modelName);
                mainWidget->setWindowTitle("Nyx - " + modelName);
                chatHistory->append("<i style='color:#00FF00;'>System: Loaded " + modelName + " (" + QString::number(checkFile.size() / (1024*1024)) + " MB) successfully.</i>");
            } else {
                chatHistory->append("<b style='color:#FF4444;'>Error:</b> GGUF Loader: File structure is not recognized as a valid model");
            }
        } catch (const std::exception& e) {
            QString errorMsg = QString::fromStdString(e.what());
            qCritical() << "Model Load Error:" << errorMsg;
            chatHistory->append("<b style='color:#FF4444;'>Error:</b> " + errorMsg);
        } catch (...) {
            qCritical() << "Model Load Error: Unknown exception";
            chatHistory->append("<b style='color:#FF4444;'>Error:</b> Unknown error during model loading");
        }
    }
});

    QObject::connect(sendButton, &QPushButton::clicked, [=, &app]() {
        QString text = inputEdit->toPlainText().trimmed();
        if (text.isEmpty()) return;

        chatHistory->append("<b>User:</b> " + text);
        inputEdit->clear();

        if (g_engine) {
            chatHistory->append("<b>AI:</b> ");
            try {
                g_engine->generate_streaming(text.toStdString(), 512, [&](const std::string& token) {
                    chatHistory->insertPlainText(QString::fromStdString(token));
                    app.processEvents(); 
                });
                chatHistory->append(""); 
            } catch (const std::exception& e) {
                chatHistory->append("<br><b style='color:red;'>Inference Error:</b> " + QString::fromStdString(e.what()));
            }
        } else {
            chatHistory->append("<b style='color:red;'>AI:</b> Please load a model first.");
        }
    });

    QObject::connect(tempSlider, &QSlider::valueChanged, [=](int v) {
        tempLabel->setText(QString("Temperature: %1").arg(v / 100.0));
    });

    mainWidget->show();
    mainWidget->raise();
    mainWidget->activateWindow();
    
    return app.exec();
}