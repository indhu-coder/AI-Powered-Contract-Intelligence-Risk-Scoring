
import { useEffect, useState } from 'react'
import { api, ApiError } from '../services/api'
import type { Contract } from '../types/api'
import { Send } from 'lucide-react'

export function ContractQApage() {
  const [contracts, setContracts] = useState<Contract[]>([])
  const [contractId, setContractId] = useState('')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] =
    useState<Awaited<ReturnType<typeof api.askQuestion>> | null>(null)
  const [loadingContracts, setLoadingContracts] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true

    async function loadContracts() {
      try {
        const result = await api.listContracts({
          page: 1,
          page_size: 100,
        })

        // Supports common paginated response field names.
       const data = result as typeof result & {
  items?: Contract[]
  results?: Contract[]
}
const rows = data.items ?? data.results ?? []

        if (active) {
          setContracts(rows)
          setError(
            rows.length ? '' : 'No contracts found. Upload a contract first.',
          )
        }
      } catch (err) {
        if (active) {
          setError(
            err instanceof ApiError
              ? err.message
              : 'Could not load contracts.',
          )
        }
      } finally {
        if (active) setLoadingContracts(false)
      }
    }

    void loadContracts()
    return () => { active = false }
  }, [])

  async function handleAsk(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    if (!contractId || !question.trim() || loading) return

    setLoading(true)
    setError('')
    setAnswer(null)

    try {
      setAnswer(await api.askQuestion(contractId, question.trim()))
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : 'Could not answer the question. Please try again.',
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="pt-6">
      <h1 className="mb-2 font-display text-3xl text-ivory">
        Contract Q&A
      </h1>
      <p className="mb-6 text-sm text-mist">
        Select a contract and ask a question about its contents.
      </p>

      <form onSubmit={handleAsk} className="ci-glass space-y-4 rounded-2xl p-5">
        <label className="block text-sm text-ivory" htmlFor="contract">
          Select contract
        </label>
        <select
          id="contract"
          value={contractId}
          onChange={(e) => {
            setContractId(e.target.value)
            setAnswer(null)
          }}
          disabled={loadingContracts || loading}
          required
          className="w-full rounded-xl border border-ink-600 bg-ink-900 p-3 text-sm text-ivory"
        >
          <option value="">
            {loadingContracts ? 'Loading contracts…' : 'Choose a contract'}
          </option>
          {contracts.map((contract) => (
            <option key={contract.id} value={contract.id}>
              {contract.filename}
            </option>
          ))}
        </select>

        <label className="block text-sm text-ivory" htmlFor="question">
          Your question
        </label>
        <textarea
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What is the expiration date?"
          maxLength={2000}
          rows={4}
          required
          className="w-full rounded-xl border border-ink-600 bg-ink-900 p-3 text-sm text-ivory"
        />

        <div className="flex justify-end">
          <button
            type="submit"
            disabled={!contractId || !question.trim() || loading || loadingContracts}
            className="inline-flex items-center gap-2 rounded-full bg-aqua px-5 py-2 text-sm font-medium text-ink-950 disabled:opacity-40"
          >
            {loading ? 'Finding answer…' : 'Ask question'}
            <Send className="h-4 w-4" />
          </button>
        </div>
      </form>

      {error && <p className="mt-4 text-sm text-coral" role="alert">{error}</p>}
      {loading && <p className="mt-4 text-sm text-mist" role="status">Analyzing the contract…</p>}

      {answer && (
        <section className="ci-glass mt-5 rounded-2xl p-5">
          <h2 className="mb-3 font-display text-xl text-ivory">Answer</h2>
          <p className="mb-3 text-sm text-mist">{answer.question}</p>
          <p className="whitespace-pre-wrap text-sm leading-7 text-ivory">
            {answer.found && answer.answer
              ? answer.answer
              : 'No reliable answer was found. Try rephrasing the question.'}
          </p>
          {answer.evidence && (
            <div className="mt-5 border-t border-ink-600 pt-4">
              <h3 className="mb-2 text-xs font-semibold uppercase text-aqua">
                Supporting evidence
              </h3>
              <p className="whitespace-pre-wrap text-sm leading-6 text-mist">
                {answer.evidence}
              </p>
            </div>
          )}
          {answer.source && (
            <p className="mt-4 text-xs text-mist/70">
              Source: {answer.source}
            </p>
          )}
        </section>
      )}
    </div>
  )
}
