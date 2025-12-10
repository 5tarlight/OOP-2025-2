package hs.ml.ui.view

import hs.ml.math.Tensor
import java.util.Scanner

class ConsoleView(private val height: Int) : MLView {
    private val scanner = Scanner(System.`in`)
    private val current = mutableListOf<String>()
    private val history = StringBuilder("[History]\n")
    private val totalRenderHeight = height + 2

    init {
        repeat(totalRenderHeight) { println() }
        render()
    }

    private object Ansi {
        const val RESET = "\u001B[0m"
        const val GRAY = "\u001B[90m"
        const val CYAN = "\u001B[36m"
        const val GREEN = "\u001B[32m"
        const val YELLOW = "\u001B[33m"
        const val CURSOR_UP = "\u001B[1A"
        const val ERASE_LINE = "\u001B[2K"
    }

    private fun render() {
        repeat(height * 2) { println() }

        println(history.toString())
        println("${Ansi.GRAY}[Scroll up to see history...]${Ansi.RESET}")
        println("${Ansi.YELLOW}------------------------------------------------------------${Ansi.RESET}")

        current.forEach {
            println(it)
        }

        val emptyLines = height - current.size
        if (emptyLines > 0) {
            repeat(emptyLines) { println() }
        }
    }

    private fun pushToHistory() {
        current.forEach {
            history.appendLine(it)
        }
    }

    override fun showMessage(message: String) {
        current.add(message)

        if (current.size > height) {
            val overflow = current.removeAt(0)
            println(overflow)
            repeat(totalRenderHeight) { println() }
            render()
        } else {
            render()
        }
    }

    private fun showBatchMessages(messages: List<String>) {
        current.addAll(messages)

        while (current.size > height) {
            val overflow = current.removeAt(0)
            println(overflow)
        }

        if (current.size <= height) {
            repeat(totalRenderHeight) { println() }
        }

        render()
    }

    override fun showError(message: String) {
        showMessage("[ERROR] $message")
    }

    override fun getInput(prompt: String?): String {
        val p = prompt ?: ""
        print("$p % ")
        val data = scanner.nextLine()

        print(Ansi.CURSOR_UP)

        println("$p $data")
        repeat(totalRenderHeight) { println() }

        pushToHistory()
        current.clear()
        render()

        return data
    }

    override fun showSingleSelectMenu(title: String, options: List<String>): Int {
        repeat(totalRenderHeight) { println() }

        val lines = mutableListOf<String>()
        lines.add("[$title]")
        options.forEachIndexed { index, option ->
            lines.add("${index + 1}. $option")
        }
        showBatchMessages(lines)

        var choice: Int
        do {
            val input = getInput()
            choice = input.toIntOrNull() ?: -1

            if (choice !in 1..options.size) {
                println("${Ansi.YELLOW}유효한 선택지가 아닙니다. 다시 시도해주세요.${Ansi.RESET}")
                Thread.sleep(1000)
                print(Ansi.CURSOR_UP)
                print(Ansi.ERASE_LINE)
                render()
            } else {
                break
            }
        } while (true)

        println("> 선택: ${options[choice - 1]}")
        repeat(totalRenderHeight) { println() }

        current.clear()
        render()

        return choice
    }

    override fun showMultiSelectMenu(title: String, options: List<String>): List<Int> {
        repeat(totalRenderHeight) { println() }

        val selectedIndices = mutableSetOf<Int>()

        while (true) {
            current.clear()

            current.add("[$title]")
            current.add("현재 선택됨: ${Ansi.CYAN}${selectedIndices.joinToString(", ") { options[it] }}${Ansi.RESET}")

            options.forEachIndexed { index, option ->
                val marker = if (index in selectedIndices) "${Ansi.GREEN}[v]${Ansi.RESET}" else "[ ]"
                current.add("$marker ${index + 1}. $option")
            }
            current.add("${options.size + 1}. 완료")

            render()

            val choice = getInput().toIntOrNull() ?: -1

            print(Ansi.CURSOR_UP)
            print(Ansi.ERASE_LINE)

            when (choice) {
                options.size + 1 -> break
                in 1..options.size -> {
                    val idx = choice - 1
                    if (idx in selectedIndices) selectedIndices.remove(idx)
                    else selectedIndices.add(idx)
                }
                else -> {}
            }
        }

        repeat(totalRenderHeight) { println() }
        current.clear()

        return selectedIndices.toList()
    }

    override fun showDataPreview(inputs: Tensor, limit: Int) {
        repeat(totalRenderHeight) { println() }

        val lines = mutableListOf<String>()
        val rows = minOf(limit, inputs.row)
        for (i in 0 until rows) {
            val rowStr = (0 until inputs.col).joinToString(", ") { j ->
                String.format("%.4f", inputs[i, j])
            }
            lines.add("[$i] $rowStr")
        }
        showBatchMessages(lines)

        getInput("엔터를 누르면 계속합니다...")

        print(Ansi.CURSOR_UP)

        repeat(totalRenderHeight) { println() }
        current.clear()
        render()
    }

    override fun showTrainingLog(epoch: Int, log: String) {
        showMessage("Epoch $epoch | $log")
    }

    override fun showEvaluationResult(index: Int, actual: Double, pred: Double) {
        val diff = pred - actual
        showMessage(
            "[$index] 실제: ${String.format("%.4f", actual)} | 예측: ${String.format("%.4f", pred)} | 오차: ${String.format("%.4f", diff)}"
        )
    }
}