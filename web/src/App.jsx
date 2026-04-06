import { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import Features from './components/Features'
import Workflow from './components/Workflow'
import Demo from './components/Demo'
import Benchmark from './components/Benchmark'
import TechStack from './components/TechStack'
import FAQ from './components/FAQ'
import QuickStart from './components/QuickStart'
import Footer from './components/Footer'

function App() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <div className="min-h-screen bg-gray-950">
      <Navbar scrolled={scrolled} />
      <main>
        <Hero />
        <Features />
        <Workflow />
        <Demo />
        <Benchmark />
        <TechStack />
        <FAQ />
        <QuickStart />
      </main>
      <Footer />
    </div>
  )
}

export default App